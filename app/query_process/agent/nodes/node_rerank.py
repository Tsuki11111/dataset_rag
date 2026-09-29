"""
重排序节点 (node_rerank)

作用：把本地知识库切片与联网搜索结果合并后统一打分重排，再用动态 Top-K 截断，
输出 reranked_docs 供答案生成使用。

三步走（与教程一致）：
1. step_1_merge_docs   —— 合并两路异构结果，统一成 {text, title, source, ...}
2. step_2_rerank_docs  —— 调重排模型给「问题-文档」对打分
3. step_3_topk         —— 按分数断崖动态截断

与教程的差异：打分改用 DashScope 重排 API（教程是本地 BGE），见 app/lm/reranker_utils.py。
"""
import sys

from app.core.logger import logger
from app.lm.reranker_utils import rerank
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_rerank"

# -----------------------------
# 动态 TopK 参数
# -----------------------------
# 硬上限：最多取前 N 条
RERANK_MAX_TOPK = 10
# 硬下限：至少保留前 N 条。
# 教程取 1，但实测出现过「首尾分差过大导致只剩 1 条上下文」的情况，
# 而答案生成只有一条切片支撑显然不够，故抬到 3。
RERANK_MIN_TOPK = 3
# 断崖阈值：相邻分数的相对跌幅超过它就截断。
#
# 教程还配了个绝对阈值 GAP_ABS=0.5，但那是按本地 BGE 的**无界 logits** 定的；
# 本项目改用的 DashScope gte-rerank-v2 返回 **0~1 归一化分数**（实测相邻最大落差
# 仅约 0.14），绝对阈值永远不会触发，是死参数，因此这里直接去掉，只保留相对阈值。
RERANK_GAP_RATIO = 0.25


def step_1_merge_docs(state: QueryGraphState) -> list:
    """
    阶段一：合并本地切片与联网结果

    两路字段结构完全不同（本地是 Milvus 切片实体，联网是标题+摘要+链接），
    统一成同一格式，其中 text 是后续打分的依据。
    """
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []
    logger.info(
        f"[{NODE_NAME}] 合并输入：本地 RRF {len(rrf_docs)} 条，联网 {len(web_docs)} 条"
    )

    doc_items = []

    # 本地切片：text 取 content
    for i, doc in enumerate(rrf_docs):
        if not isinstance(doc, dict):
            logger.warning(f"[{NODE_NAME}] 跳过结构异常的本地文档(index={i})：{type(doc).__name__}")
            continue
        content = (doc.get("content") or "").strip()
        if not content:
            continue
        doc_id = doc.get("chunk_id") or doc.get("id")
        doc_items.append({
            "text": content,
            "doc_id": doc_id,
            "chunk_id": doc_id,
            "title": doc.get("title") or doc.get("item_name") or "",
            "url": "",
            "source": "local",
        })

    # 联网结果：text 取 snippet，天然没有切片主键
    for i, doc in enumerate(web_docs):
        if not isinstance(doc, dict):
            logger.warning(f"[{NODE_NAME}] 跳过结构异常的联网结果(index={i})：{type(doc).__name__}")
            continue
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        if not text:
            continue
        doc_items.append({
            "text": text,
            "doc_id": None,
            "chunk_id": None,
            "title": (doc.get("title") or "").strip(),
            "url": (doc.get("url") or "").strip(),
            "source": "web",
        })

    logger.info(f"[{NODE_NAME}] 合并完成，共 {len(doc_items)} 条待打分")
    return doc_items


def step_2_rerank_docs(state: QueryGraphState, doc_items: list) -> list:
    """
    阶段二：调重排模型打分并按分数降序

    重排失败时降级为原始顺序（分数记 0），保证链路不中断。
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""
    if not doc_items or not question:
        logger.warning(f"[{NODE_NAME}] 无文档或无问题，跳过重排")
        return []

    texts = [d["text"] for d in doc_items]
    try:
        scored = rerank(question, texts)
    except Exception as e:
        logger.error(f"[{NODE_NAME}] 重排失败，降级为原始顺序：{e}", exc_info=True)
        return [{**d, "score": 0.0} for d in doc_items]

    # 接口按 index 指回入参下标，这里还原成完整文档
    out = [{**doc_items[s["index"]], "score": s["score"]} for s in scored]
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def step_3_topk(scored_docs: list) -> list:
    """
    阶段三：动态 Top-K

    不用机械的「取前 N 条」，而是在 [MIN_TOPK, MAX_TOPK] 区间内找分数断崖：
    相邻两条落差过大说明相关性骤降，就在那里截断，避免低分文档混入候选。
    """
    if not scored_docs:
        return []

    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    topk = max_topk  # 没触发断崖就取满上限

    if topk > RERANK_MIN_TOPK:
        # 从 MIN_TOPK 之后开始探测相邻落差（索引从 0 起，故起点为 MIN_TOPK-1），
        # 这样 MIN_TOPK 就是硬地板，断崖再陡也不会把上下文截到它以下
        for i in range(RERANK_MIN_TOPK - 1, max_topk - 1):
            s1 = scored_docs[i].get("score") or 0.0
            s2 = scored_docs[i + 1].get("score") or 0.0
            gap = s1 - s2  # 已降序，gap 恒 >= 0
            rel = gap / (abs(s1) + 1e-6)  # 1e-6 防除零
            if rel >= RERANK_GAP_RATIO:
                logger.info(
                    f"[{NODE_NAME}] 触发断崖截断 @ 第 {i + 1} 条 "
                    f"(score {s1:.4f} -> {s2:.4f}, gap={gap:.4f}, rel={rel:.3f})"
                )
                topk = i + 1
                break

    return scored_docs[:topk]


def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 重排序 (node_rerank)

    :param state: 需包含 session_id，以及 rrf_chunks / web_search_docs 至少一路
    :return: {"reranked_docs": [带 score 的文档]}；无有效输入返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    try:
        doc_items = step_1_merge_docs(state)
        scored_docs = step_2_rerank_docs(state, doc_items)
        topk_docs = step_3_topk(scored_docs)

        logger.info(f"[{NODE_NAME}] [{function_name}] 最终输出 {len(topk_docs)} 条")
        for rank, d in enumerate(topk_docs, 1):
            logger.info(
                f"[{NODE_NAME}] [{function_name}]   {rank}. "
                f"[{d['source']}] score={d['score']:.4f} "
                f"{d['title'][:36]!r}"
            )
        return {"reranked_docs": topk_docs}

    except Exception as e:
        # 重排失败不中断链路：返回空结果，答案生成会拿到空上下文
        logger.error(f"[{NODE_NAME}] [{function_name}] 重排异常：{e}", exc_info=True)
        return {"reranked_docs": []}
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")


if __name__ == '__main__':
    """
    本地测试：用伪造的两路结果走完整流程（会真实调用重排接口）

    重点验证：两路都进了合并、输出按分数降序、不超过硬上限、本地与联网都在结果里。
    """
    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    rrf_chunks = [
        {"chunk_id": 101, "title": "## 装入全幅烫金膜盒",
         "content": "打开烫金膜盒支架盖，将烫金膜盒装入并向下轻推，直到它锁定到位。"},
        {"chunk_id": 102, "title": "## 更换电池",
         "content": "关闭设备并拔掉电源适配器，卸下后盖后更换电池。"},
        {"chunk_id": 103, "title": "## 使用设备",
         "content": "使用设备前确保已准备好所需材料并阅读安全须知。"},
    ]
    web_search_docs = [
        {"title": "HAK180 快速设置指导手册", "url": "https://example.com/a",
         "snippet": "烫金膜盒的安装步骤：先打开支架盖，再沿导轨推入并压紧。"},
        {"title": "烫金机保养常识", "url": "https://example.com/b",
         "snippet": "每日需检查电源与润滑部位，定期清理膜屑。"},
    ]

    session_id = "rerank_test"
    st = create_query_default_state(
        session_id=session_id,
        original_query="HAK 180 烫金机怎么安装烫金膜盒？",
        rewritten_query="HAK 180 烫金机怎么安装烫金膜盒？",
        is_stream=False,
        rrf_chunks=rrf_chunks,
        web_search_docs=web_search_docs,
    )

    try:
        # 合并阶段单独验证：动态 TopK 会截断，不能拿最终输出去判断合并有没有丢源
        merged = step_1_merge_docs(st)
        merged_sources = {d["source"] for d in merged}
        logger.info(f"[测试] 合并阶段 {len(merged)} 条，来源={merged_sources}")

        result = node_rerank(st)
        got = result.get("reranked_docs") or []
        logger.info(f"[测试] 最终输出 {len(got)} 条")
        for rank, d in enumerate(got, 1):
            logger.info(f"[测试]   {rank}. [{d['source']}] {d['score']:.4f} {d['title'][:32]!r}")

        problems = []
        if len(merged) != len(rrf_chunks) + len(web_search_docs):
            problems.append(f"合并条数不符：{len(merged)} != {len(rrf_chunks) + len(web_search_docs)}")
        if merged_sources != {"local", "web"}:
            problems.append(f"合并阶段丢失了某一路结果，来源={merged_sources}")

        if not got:
            problems.append("没有输出任何文档")
        if len(got) > RERANK_MAX_TOPK:
            problems.append(f"超过硬上限：{len(got)} > {RERANK_MAX_TOPK}")
        scores = [d["score"] for d in got]
        if scores != sorted(scores, reverse=True):
            problems.append("输出未按分数降序")
        # 每条结果的 text 都应能在原始输入里找到出处
        known_texts = {d["content"] for d in rrf_chunks} | {d["snippet"] for d in web_search_docs}
        if any(d["text"] not in known_texts for d in got):
            problems.append("存在 text 来源不明的结果")

        for p in problems:
            logger.error(f"[测试] [FAIL] {p}")
        if not problems:
            logger.success("[测试] [PASS] 重排节点验证通过")
    except Exception as e:
        logger.error(f"[测试] [FAIL] 执行失败：{e}", exc_info=True)
    finally:
        clear_task(session_id)
