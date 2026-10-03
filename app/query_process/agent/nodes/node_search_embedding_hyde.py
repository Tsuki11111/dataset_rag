"""
HyDE 检索节点 (node_search_embedding_hyde)

HyDE (Hypothetical Document Embeddings)：先让大模型"脑补"一段可能的答案，
再用「改写后的问题 + 假设文档」去检索。

为什么有效：用户的问题通常很短（"烫金膜盒怎么安装"），而知识库里是成段的说明书正文，
短查询与长文档在语义空间里天然不匹配。假设文档由大模型写成"说明书的样子"，
向量更接近真实切片，召回率因此提升。

代价：假设文档可能"脑补过头"带偏方向，所以本项目把它作为**多路召回之一**，
最终由 RRF 融合 + 重排来纠偏，不单独依赖它。

与教程的差异：教程用稠密+稀疏混合检索，本项目嵌入模型只输出稠密向量（见
node_search_embedding 的说明），因此走 dense_search。
"""
import sys

from langchain.messages import HumanMessage

from app.clients.milvus_utils import dense_search, get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.error_policy import degrade, degrade_dependency
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.escape_milvus_string_utils import build_item_name_filter
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_search_embedding_hyde"

# 本路检索返回的切片数量
TOP_K = 5
# HNSW 检索参数 ef：越大召回越准、越慢
SEARCH_EF = 64

# 检索时取回的业务字段（与 node_search_embedding 保持一致，供下游重排与答案生成使用）
OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title", "file_title", "item_name"]


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    步骤 1: 用大模型生成假设性文档

    :param rewritten_query: 改写后的用户问题
    :return: 假设文档文本；生成失败返回空字符串（调用方据此降级）
    """
    function_name = sys._getframe().f_code.co_name
    if not rewritten_query:
        logger.error(f"[{NODE_NAME}] [{function_name}] rewritten_query 为空，无法生成假设文档")
        return ""

    try:
        llm = get_llm_client()
        prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        response = llm.invoke([HumanMessage(content=prompt)])
        hyde_doc = (response.content or "").strip()
        logger.info(f"[{NODE_NAME}] [{function_name}] 假设文档生成完成，长度={len(hyde_doc)}")
        logger.debug(f"[{NODE_NAME}] [{function_name}] 假设文档预览：{hyde_doc[:120]!r}")
        return hyde_doc
    except Exception as e:
        # HyDE 只是四路召回之一，缺了这一路不影响其他路——但仍区分「外部故障」与「代码写错」
        return degrade(NODE_NAME, "生成假设文档", "", e)


def step_2_search_by_hyde(rewritten_query: str, hyde_doc: str, item_names) -> list:
    """
    步骤 2: 用「改写问题 + 假设文档」检索 kb_chunks

    把问题与假设文档拼在一起做向量化：问题负责"问的是什么"，
    假设文档负责补充"可能怎么回答"，合起来的语义比单独用问题更饱满。

    :return: 命中切片列表；失败返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    combined_text = f"{rewritten_query} {hyde_doc}".strip()
    logger.info(f"[{NODE_NAME}] [{function_name}] 拼接查询+假设文档，长度={len(combined_text)}")

    try:
        embeddings = generate_embeddings([combined_text])
        dense_vectors = (embeddings or {}).get("dense") or []
        if not dense_vectors:
            logger.error(f"[{NODE_NAME}] [{function_name}] 向量化失败，返回空结果")
            return []

        expr = build_item_name_filter(item_names)
        if expr:
            logger.info(f"[{NODE_NAME}] [{function_name}] 过滤条件：{expr}")
        else:
            # 正常流程里 item_names 不应为空（图只在确认产品后才路由到这里）
            logger.warning(f"[{NODE_NAME}] [{function_name}] item_names 为空，退化为全库检索")

        client = get_milvus_client()
        if client is None:
            return degrade_dependency(NODE_NAME, "假设文档检索", [], "Milvus 不可用")
            return []

        res = dense_search(
            client=client,
            collection_name=milvus_config.chunks_collection,
            dense_vector=dense_vectors[0],
            limit=TOP_K,
            expr=expr or None,
            output_fields=OUTPUT_FIELDS,
            search_params={"ef": SEARCH_EF},
        )
        return res[0] if res else []

    except Exception as e:
        return degrade(NODE_NAME, "假设文档向量检索", [], e)


def node_search_embedding_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 假设性文档检索 (node_search_embedding_hyde)

    流程：
    1. 取改写后的问题（无则回退原始问题）
    2. 大模型生成假设性文档
    3. 用「问题 + 假设文档」在 kb_chunks 中执行稠密检索
    4. 返回 hyde_embedding_chunks 与 hyde_doc

    :param state: 需包含 session_id / rewritten_query / item_names
    :return: {"hyde_embedding_chunks": [命中切片], "hyde_doc": 假设文档}
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    try:
        # 1. 取查询文本
        rewritten_query = state.get("rewritten_query") or state.get("original_query")
        if not rewritten_query:
            logger.error(f"[{NODE_NAME}] [{function_name}] 无有效查询，返回空结果")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}

        item_names = state.get("item_names") or []
        logger.info(f"[{NODE_NAME}] [{function_name}] 入参：query={rewritten_query!r}，item_names={item_names}")

        # 2. 生成假设文档
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
        if not hyde_doc:
            logger.warning(f"[{NODE_NAME}] [{function_name}] 假设文档为空，跳过本路检索")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}

        # 3. 检索
        chunks = step_2_search_by_hyde(rewritten_query, hyde_doc, item_names)
        logger.info(f"[{NODE_NAME}] [{function_name}] 检索完成，召回 {len(chunks)} 条切片")
        if chunks:
            top1 = chunks[0]
            logger.info(
                f"[{NODE_NAME}] [{function_name}] Top1："
                f"相似度={top1.get('distance', 0):.4f}，"
                f"标题={(top1.get('entity') or {}).get('title', '')[:40]!r}"
            )

        return {"hyde_embedding_chunks": chunks, "hyde_doc": hyde_doc}

    except Exception as e:
        return degrade(NODE_NAME, "HyDE 检索", {"hyde_embedding_chunks": [], "hyde_doc": ""}, e)
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")


if __name__ == '__main__':
    """
    本地测试：验证假设文档生成与检索

    前置：Milvus 与 LLM 可用，kb_chunks 中已有数据
    """
    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    cases = [
        ("带产品名过滤", "烫金膜盒怎么安装？", ["Brother HAK 180 烫金机"]),
        ("不带过滤（全库）", "怎么更换电池？", []),
    ]

    for label, query, item_names in cases:
        logger.info("=" * 70)
        logger.info(f"[测试] {label}：query={query!r}，item_names={item_names}")
        st = create_query_default_state(
            session_id=f"hyde_test_{label}",
            original_query=query,
            rewritten_query=query,
            item_names=item_names,
            is_stream=False,
        )
        try:
            result = node_search_embedding_hyde(st)
            doc = result.get("hyde_doc") or ""
            chunks = result.get("hyde_embedding_chunks") or []
            logger.info(f"[测试] 假设文档（{len(doc)}字）：{doc[:110]!r}")
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
