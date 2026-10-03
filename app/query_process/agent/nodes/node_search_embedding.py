"""
基线向量检索节点 (node_search_embedding)

作用：把改写后的用户问题向量化，在 kb_chunks 里召回最相关的切片。

与教程的差异：教程用「稠密+稀疏」混合检索（BGE-M3 双向量），
本项目嵌入模型为 DashScope text-embedding-v2，只输出稠密向量，
因此走 dense_search 单路检索，不再有加权融合。
"""
import sys

from app.clients.milvus_utils import dense_search, get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.error_policy import degrade, degrade_dependency
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.state import QueryGraphState
from app.utils.escape_milvus_string_utils import build_item_name_filter
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_search_embedding"

# 本路检索返回的切片数量（后续由 RRF 融合、重排再进一步筛选）
TOP_K = 5
# HNSW 检索参数 ef：越大召回越准、越慢
SEARCH_EF = 64

# 检索时取回的业务字段：
# - content：下游重排要拿它和问题一起打分
# - title / parent_title / file_title：生成答案时补上下文、标注来源
# - chunk_id：主键，便于后续按 id 回查
OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title", "file_title", "item_name"]


def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 向量检索 (node_search_embedding)

    流程：
    1. 取改写后的问题（无则回退原始问题）
    2. 向量化
    3. 构造产品名过滤表达式（限定只在该产品的切片里检索）
    4. 在 kb_chunks 中执行稠密检索
    5. 返回 embedding_chunks

    :param state: 需包含 session_id / rewritten_query / item_names
    :return: {"embedding_chunks": [命中切片]}；失败或无结果返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    try:
        # 1. 取查询文本：优先用改写后的问题（含产品名、指代已消解）
        query = state.get("rewritten_query") or state.get("original_query")
        if not query:
            logger.error(f"[{NODE_NAME}] [{function_name}] 无有效查询（rewritten_query 与 original_query 均为空）")
            return {"embedding_chunks": []}

        item_names = state.get("item_names") or []
        logger.info(f"[{NODE_NAME}] [{function_name}] 入参：query={query!r}，item_names={item_names}")

        # 2. 向量化
        embeddings = generate_embeddings([query])
        dense_vectors = (embeddings or {}).get("dense") or []
        if not dense_vectors:
            logger.error(f"[{NODE_NAME}] [{function_name}] 查询向量化失败，返回空结果")
            return {"embedding_chunks": []}

        # 3. 产品名过滤：限定在已确认产品的切片里检索，避免串到别的产品
        expr = build_item_name_filter(item_names)
        if expr:
            logger.info(f"[{NODE_NAME}] [{function_name}] 过滤条件：{expr}")
        else:
            # 正常流程里 item_names 不应为空（图只在确认产品后才路由到这里），
            # 走到这说明上游异常，此时退化为全库检索并告警，而不是静默返回空
            logger.warning(f"[{NODE_NAME}] [{function_name}] item_names 为空，退化为全库检索")

        # 4. 稠密检索
        client = get_milvus_client()
        if client is None:
            return degrade_dependency(NODE_NAME, "向量检索", {"embedding_chunks": []}, "Milvus 不可用")
            return {"embedding_chunks": []}

        res = dense_search(
            client=client,
            collection_name=milvus_config.chunks_collection,
            dense_vector=dense_vectors[0],
            limit=TOP_K,
            expr=expr or None,
            output_fields=OUTPUT_FIELDS,
            search_params={"ef": SEARCH_EF},
        )

        # dense_search 返回「每条查询向量的结果列表」，这里只有一条查询，取 res[0]
        chunks = res[0] if res else []
        logger.info(f"[{NODE_NAME}] [{function_name}] 检索完成，召回 {len(chunks)} 条切片")
        if chunks:
            top1 = chunks[0]
            logger.info(
                f"[{NODE_NAME}] [{function_name}] Top1："
                f"相似度={top1.get('distance', 0):.4f}，"
                f"标题={(top1.get('entity') or {}).get('title', '')[:40]!r}"
            )

        return {"embedding_chunks": chunks}

    except Exception as e:
        # 向量检索是四路召回之一：失败就降级（RRF 会忽略空路），
        # 但编程错误会被 degrade 上抛——正是这类错误此前被静默吞掉过
        return degrade(NODE_NAME, "向量检索", {"embedding_chunks": []}, e)
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")


if __name__ == '__main__':
    """
    本地测试：验证带过滤与不带过滤两种检索

    前置：Milvus 已启动，kb_chunks 中已有数据
    """
    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    cases = [
        ("带产品名过滤", "烫金膜盒怎么安装？", ["Brother HAK 180 烫金机"]),
        ("产品名不存在", "烫金膜盒怎么安装？", ["不存在的产品"]),
        ("不带过滤（全库）", "怎么更换电池？", []),
    ]

    for label, query, item_names in cases:
        logger.info("=" * 70)
        logger.info(f"[测试] {label}：query={query!r}，item_names={item_names}")
        st = create_query_default_state(
            session_id=f"search_test_{label}",
            original_query=query,
            rewritten_query=query,
            item_names=item_names,
            is_stream=False,
        )
        try:
            result = node_search_embedding(st)
            chunks = result.get("embedding_chunks") or []
            logger.info(f"[测试] 召回 {len(chunks)} 条")
            for h in chunks[:3]:
                ent = h.get("entity") or {}
                logger.info(
                    f"[测试]   {h.get('distance', 0):.4f}  {ent.get('item_name')}  "
                    f"{(ent.get('title') or '')[:34]}"
                )
        except Exception as e:
            logger.error(f"[测试] 执行失败：{e}", exc_info=True)
        finally:
            clear_task(st["session_id"])

    logger.info("=" * 70)
    logger.info("[测试] 全部用例执行完毕")
