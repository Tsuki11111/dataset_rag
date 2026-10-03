"""
图谱查询节点 (node_query_kg)

作用：把用户问题里提到的实体拿去查 Neo4j，取回相关切片，作为四路召回中的「图谱」一路。

与向量检索的区别：向量检索靠语义相似，图谱靠**实体与实体间关系**做结构化召回，
能捞到语义上并不相似、但通过关系关联的切片（多跳）。

**输出必须是切片形状**（含 `chunk_id` + `content`）才能流经 RRF → 重排 → 答案。
而图谱的 `:Chunk` 节点只存 chunk_id 与标题、**不存正文**——正文留在 Milvus，
所以这里要按 chunk_id 回 Milvus 取（`fetch_chunks_by_chunk_ids` 正是为这个场景准备的）。
这样图谱不必重复存一份文本。
"""
import sys

from app.clients.milvus_utils import fetch_chunks_by_chunk_ids, get_milvus_client
from app.clients.neo4j_utils import is_neo4j_available, query_kg_chunks
from app.conf.milvus_config import milvus_config
from app.core.error_policy import degrade, degrade_dependency
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_query_kg"

# 图谱这一路最多贡献多少切片。
# 与向量那两路保持一致（都是 5）——图谱给太多会挤占 RRF 的合并名额，
# 而且图谱侧的排序只是近似（按实体加权），真正的精度由下游重排决定。
KG_TOP_K = 5

# 回 Milvus 取正文时要哪些字段，与 node_search_embedding 保持一致
OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title", "file_title", "item_name"]


def step_1_get_inputs(state: QueryGraphState):
    """步骤 1: 取检索词与已确认的产品名"""
    function_name = sys._getframe().f_code.co_name
    query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names") or []
    logger.info(f"[{NODE_NAME}] [{function_name}] 入参：query={query!r}，item_names={item_names}")
    return query, item_names


def step_2_query_graph(item_names: list, query: str) -> list:
    """步骤 2: 查图谱，拿到相关切片 id（按相关度降序）"""
    return query_kg_chunks(item_names, query, limit=KG_TOP_K)


def step_3_fetch_content(hits: list) -> list:
    """
    步骤 3: 按 chunk_id 回 Milvus 取正文，组装成切片形状

    保持图谱给的顺序（相关度降序），Milvus 返回的顺序不保证一致。
    """
    function_name = sys._getframe().f_code.co_name
    client = get_milvus_client()
    if client is None:
        return degrade_dependency(NODE_NAME, "图谱切片正文取回", [], "Milvus 不可用")

    rows = fetch_chunks_by_chunk_ids(
        client,
        milvus_config.chunks_collection,
        [h["chunk_id"] for h in hits],
        output_fields=OUTPUT_FIELDS,
    )
    by_id = {str(r["chunk_id"]): r for r in rows}

    docs = []
    for hit in hits:
        row = by_id.get(hit["chunk_id"])
        if row is None:
            # 图谱里有、Milvus 里没有：该切片被重新导入过（chunk_id 会变新），图谱尚未跟上
            logger.warning(
                f"[{NODE_NAME}] [{function_name}] 切片 {hit['chunk_id']} 在 Milvus 中不存在，已跳过"
            )
            continue
        docs.append({
            "chunk_id": str(row["chunk_id"]),
            "content": row.get("content") or "",
            "title": row.get("title") or "",
            "parent_title": row.get("parent_title") or "",
            "file_title": row.get("file_title") or "",
            "item_name": row.get("item_name") or "",
            # 图谱侧的相关度与来源实体，留着便于排查「这条为什么被召回」
            "kg_score": hit["score"],
            "kg_via": hit["via"],
        })
    return docs


def node_query_kg(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 图谱查询 (node_query_kg)

    :param state: 需包含 session_id / rewritten_query / item_names
    :return: {"kg_chunks": [切片实体]}；无命中、缺入参或 Neo4j 不可用时返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    try:
        query, item_names = step_1_get_inputs(state)
        if not query or not item_names:
            logger.warning(f"[{NODE_NAME}] [{function_name}] 缺少问题或产品名，跳过图谱检索")
            return {"kg_chunks": []}

        # 前置检查：Neo4j 不可用就整段跳过。图谱是补充召回，不该拖垮整条链路
        if not is_neo4j_available():
            return degrade_dependency(NODE_NAME, "图谱检索", {"kg_chunks": []}, "Neo4j 不可用")

        hits = step_2_query_graph(item_names, query)
        if not hits:
            logger.info(f"[{NODE_NAME}] [{function_name}] 图谱无命中，返回空结果")
            return {"kg_chunks": []}

        docs = step_3_fetch_content(hits)
        logger.info(f"[{NODE_NAME}] [{function_name}] 图谱召回 {len(docs)} 条切片")
        if docs:
            logger.info(
                f"[{NODE_NAME}] [{function_name}] Top1："
                f"score={docs[0]['kg_score']}，"
                f"经由实体={docs[0]['kg_via']}，"
                f"标题={docs[0]['title'][:36]!r}"
            )
        return {"kg_chunks": docs}

    except Exception as e:
        # 图谱是四路召回之一，失败降级；编程错误由 degrade 上抛
        return degrade(NODE_NAME, "图谱检索", {"kg_chunks": []}, e)
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")


if __name__ == '__main__':
    """
    本地测试：用真实图谱验证召回

    前置：Neo4j 与 Milvus 都在运行，且库里已有图谱数据
    """
    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    PRODUCT = "Brother HAK 180 烫金机"
    cases = [
        ("命中（问题含实体名）", "Brother HAK 180 烫金机怎么安装烫金膜盒？", [PRODUCT]),
        ("问题不含实体名", "今天天气怎么样", [PRODUCT]),
        ("产品名不在图谱里", "烫金膜盒怎么安装？", ["不存在产品"]),
        ("缺产品名", "烫金膜盒怎么安装？", []),
    ]

    for label, query, item_names in cases:
        session_id = f"kg_query_test_{label}"
        logger.info("=" * 70)
        logger.info(f"[测试] {label}：query={query!r}，item_names={item_names}")
        st = create_query_default_state(
            session_id=session_id,
            original_query=query,
            rewritten_query=query,
            item_names=item_names,
            is_stream=False,
        )
        try:
            docs = node_query_kg(st).get("kg_chunks") or []
            logger.info(f"[测试] 召回 {len(docs)} 条")
            for i, d in enumerate(docs[:3], 1):
                logger.info(
                    f"[测试]   {i}. score={d['kg_score']} "
                    f"经由={d['kg_via']} {d['title'][:30]!r}"
                )
            if label.startswith("命中") and not docs:
                logger.error("[测试] [FAIL] 本该命中却没有结果")
            if not label.startswith("命中") and docs:
                logger.error(f"[测试] [FAIL] 本该返回空却有 {len(docs)} 条")
        except Exception as e:
            logger.error(f"[测试] [FAIL] 执行失败：{e}", exc_info=True)
        finally:
            clear_task(session_id)

    logger.info("=" * 70)
    logger.info("[测试] 全部用例执行完毕")
