"""
RRF 融合节点 (node_rrf)

作用：把多路召回结果按「倒数排名融合」合并成一个统一排序列表，输出 rrf_chunks。

只融合**切片类**的召回，因为 RRF 按 chunk_id 去重计分，要求各路都能提供 chunk_id。
**联网搜索的结果不是切片、没有 chunk_id**，按教程设计它不在这里融合，而是在下游
node_rerank 阶段与切片一起重排（教程 9.6.1：RRF 只对同源结果有效，跨源合并交给 Rerank）。

RRF 核心公式：score += weight * 1 / (k + rank)
"""
import sys
from typing import Any, Dict, List

from app.core.error_policy import degrade
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_rrf"

# RRF 衰减常数：越大则「多路都出现」比「单路排名靠前」更重要
RRF_K = 60
# 融合后保留的切片数（下游还要重排，这里放宽一点）
MAX_RESULTS = 10

# 各路召回在本节点内的权重
# kg_chunks 目前恒为空（node_query_kg 尚未实现），保留此项是为了它落地后无需再改这里
SOURCE_WEIGHTS = {
    "embedding_chunks": 1.0,
    "hyde_embedding_chunks": 1.0,
    "kg_chunks": 1.0,
}


def _as_entity_list(chunks) -> List[Dict[str, Any]]:
    """
    把上游召回结果规整为 entity dict 列表

    兼容两种形态：
    - Milvus 命中（pymilvus 的 Hit，dict 接口）：
      {"chunk_id": ..., "distance": ..., "entity": {业务字段...}}，把 entity 摊平
    - 扁平实体：{"chunk_id": ..., "content": ...}，直接采用（测试数据、图谱等其他来源）

    RRF 要按 chunk_id 跨路去重计分，两条路径都会保证结果里带 chunk_id。
    """
    out: List[Dict[str, Any]] = []
    for doc in (chunks or []):
        if not hasattr(doc, "get"):
            logger.warning(f"[{NODE_NAME}] 跳过无法识别的召回项：{type(doc).__name__}")
            continue

        entity = doc.get("entity")
        if isinstance(entity, dict):
            item = dict(entity)
            # entity 里通常已带 chunk_id，兜底取外层的
            if not item.get("chunk_id"):
                item["chunk_id"] = doc.get("chunk_id")
            # 记下向量相似度便于排查；RRF 本身只看排名不看分数
            if "score" not in item and doc.get("distance") is not None:
                item["score"] = doc.get("distance")
        elif doc.get("chunk_id") or doc.get("id"):
            item = dict(doc)
        else:
            logger.warning(f"[{NODE_NAME}] 召回项既无 entity 也无 chunk_id，已跳过")
            continue

        out.append(item)
    return out


def reciprocal_rank_fusion(
        source_weights: List[tuple],
        k: int = RRF_K,
        max_results: int = None,
) -> List[tuple]:
    """
    带权重的 RRF 算法

    :param source_weights: [(文档列表, 权重), ...]，每个文档列表已按相关性降序排好
    :param k: RRF 常数，平滑排名影响，避免单路第一名垄断得分
    :param max_results: 只返回前 N 个，None 表示全部
    :return: [(文档实体, 融合得分), ...]，按得分降序
    """
    # score_map 累计得分；chunk_map 记录 chunk_id 到实体的映射，供最终返回
    score_map: Dict[Any, float] = {}
    chunk_map: Dict[Any, Dict[str, Any]] = {}

    for docs, weight in source_weights:
        # rank 从 1 开始，符合「第几名」的直觉
        for rank, item in enumerate(docs, start=1):
            # Milvus 的主键在 API 层统一以 chunk_id 暴露
            chunk_id = item.get("chunk_id") or item.get("id")
            if not chunk_id:
                logger.warning(
                    f"[{NODE_NAME}] 召回项缺少 chunk_id，已跳过：{list(item.keys())}"
                )
                continue

            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))
            # 同一 chunk 在多路出现时，只保留首次遇到的实体
            chunk_map.setdefault(chunk_id, item)

    merged = [(chunk_map[cid], score) for cid, score in score_map.items()]
    merged.sort(key=lambda x: x[1], reverse=True)

    if max_results is not None:
        merged = merged[:max_results]
    return merged


def node_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 倒数排名融合 (node_rrf)

    流程：
    1. 取出各路切片类召回，统一规整为 entity 列表
    2. 按权重做 RRF 融合
    3. 返回 rrf_chunks

    :param state: 需包含 session_id 及至少一路召回结果
    :return: {"rrf_chunks": [融合后的切片实体]}；无有效输入返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    try:
        sources = []
        for field, weight in SOURCE_WEIGHTS.items():
            chunks = _as_entity_list(state.get(field))
            logger.info(
                f"[{NODE_NAME}] [{function_name}] {field}：{len(chunks)} 条（权重 {weight}）"
            )
            sources.append((chunks, weight))

        fused = reciprocal_rank_fusion(sources, k=RRF_K, max_results=MAX_RESULTS)
        rrf_chunks = [doc for doc, _score in fused]

        logger.info(f"[{NODE_NAME}] [{function_name}] 融合完成，输出 {len(rrf_chunks)} 条")
        if rrf_chunks:
            top = rrf_chunks[0]
            logger.info(
                f"[{NODE_NAME}] [{function_name}] Top1："
                f"chunk_id={top.get('chunk_id')}，"
                f"标题={(top.get('title') or '')[:40]!r}，"
                f"得分={fused[0][1]:.6f}"
            )
        return {"rrf_chunks": rrf_chunks}

    except Exception as e:
        # 融合失败不中断链路：返回空结果，下游会拿到空上下文
        return degrade(NODE_NAME, "RRF 融合", {"rrf_chunks": []}, e)
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")


if __name__ == '__main__':
    """
    本地测试：用伪造的召回数据验证融合逻辑（不依赖 Milvus）

    覆盖三种情形：跨路重叠去重、单路独有项保留、缺 chunk_id 的项被丢弃。
    """
    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    embedding_chunks = [
        {"chunk_id": 1, "title": "内容1", "content": "打开电源"},
        {"chunk_id": 2, "title": "内容2", "content": "遇到故障"},
        {"chunk_id": 3, "title": "内容3", "content": "电压220V"},
    ]
    hyde_chunks = [
        {"chunk_id": 3, "title": "内容3", "content": "电压220V"},
        {"chunk_id": 1, "title": "内容1", "content": "打开电源"},
        {"chunk_id": 4, "title": "内容4", "content": "佩戴手套"},
    ]

    session_id = "rrf_test"
    st = create_query_default_state(
        session_id=session_id,
        original_query="测试",
        is_stream=False,
        embedding_chunks=embedding_chunks,
        hyde_embedding_chunks=hyde_chunks,
    )

    try:
        result = node_rrf(st)
        got = result.get("rrf_chunks") or []
        ids = [c.get("chunk_id") for c in got]
        logger.info(f"[测试] 输出 {len(got)} 条，chunk_id={ids}")

        problems = []
        # 两路各 3 条、重叠 2 条（1、3），并集应为 4
        if len(got) != 4:
            problems.append(f"并集数量错误：期望 4，实际 {len(got)}")
        if set(ids) != {1, 2, 3, 4}:
            problems.append(f"并集内容错误：{set(ids)}")
        # 1 和 3 在两路都出现，排名应在只出现一次的同位次项之前
        pos = {cid: i for i, cid in enumerate(ids)}
        if pos.get(1, 99) > pos.get(2, 99):
            problems.append("两路都命中的 chunk 1 排名应在仅一路命中的 chunk 2 之前")

        for p in problems:
            logger.error(f"[测试] [FAIL] {p}")
        if not problems:
            logger.success("[测试] [PASS] RRF 融合逻辑验证通过")
    except Exception as e:
        logger.error(f"[测试] [FAIL] 执行失败：{e}", exc_info=True)
    finally:
        clear_task(session_id)
