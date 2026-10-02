"""
答案生成节点 (node_answer_output)

职责：
1. 前置节点已产出 answer（产品名反问 / 拒识）时直接输出，跳过 LLM 生成
2. 否则用 `reranked_docs` 作参考内容调大模型生成答案
3. 流式模式下逐块推送 delta，结束后推送 final（含解析出的图片 URL）
4. 答案存档到 MongoDB，供 /history 接口读取

**图片的处理**：`prompts/answer_out.prompt` 要求模型在答案末尾追加一个【图片】区块，
本节点把它拆出来单独作为 `images` 返回（每项含 `url` 与图注），并从正文里去掉
（用户不必看到一堆裸链接）。图注取自该图所在切片的标题。

**只放行参考内容里真实出现过的 URL** —— 模型可能编造或改写链接，
直接透传会让前端显示一排破图。
"""
import re
import sys

from langchain.messages import HumanMessage, SystemMessage

from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_answer_output"

# 带进提示词的历史消息条数
HISTORY_LIMIT = 6
# 单个切片拼进上下文时的上限，防止个别超长切片吃光窗口
MAX_CONTEXT_CHARS_PER_DOC = 1200
# 图片区块标记，与 prompts/answer_out.prompt 里的约定一致
IMAGE_MARKER = "【图片】"
# 从切片正文里抓 Markdown 图片链接 ![](url)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
# 一片都没检索到时的兜底答复（此时调 LLM 只会让它凭空编）
FALLBACK_ANSWER = "抱歉，没有检索到与该问题相关的内容。可以换个说法，或确认一下产品型号。"

SYSTEM_PROMPT = "你是产品使用文档的问答助手。回答必须严格基于提供的参考内容，不确定就说明没有找到，不要编造。"


def _build_context(docs: list):
    """
    把重排后的切片拼成参考内容，同时记下每个图片 URL 出自哪条切片

    返回的 captions 一举两得：既当**白名单**（放行只在它里面出现过的 URL），
    又给前端提供**图注**（用来源切片的标题）。

    :return: (context 文本, {url: 图注})
    """
    parts, captions = [], {}
    for i, doc in enumerate(docs, 1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        # 切片标题形如 "## 3.4.2 装入半幅烫金膜盒"，去掉井号当图注更干净
        title = (doc.get("title") or "").strip().lstrip("#").strip()
        parts.append(f"[{i}] {title}\n{text[:MAX_CONTEXT_CHARS_PER_DOC]}")
        for url in MARKDOWN_IMAGE_RE.findall(text):
            captions.setdefault(url, title)
    return "\n\n".join(parts), captions


def _build_history(session_id: str) -> str:
    """把最近几轮对话拼成文本，供模型理解指代"""
    function_name = sys._getframe().f_code.co_name
    try:
        msgs = get_recent_messages(session_id, limit=HISTORY_LIMIT)
    except Exception as e:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 读取历史失败，将无历史继续：{e}")
        return "（无）"

    # 本轮问题已由 node_item_name_confirm 存入历史，去掉以免与【用户问题】重复
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]

    lines = []
    for m in msgs:
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{'用户' if m.get('role') == 'user' else '助手'}：{content}")
    return "\n".join(lines) if lines else "（无）"


def _split_images(text: str, captions: dict):
    """
    拆出答案末尾的【图片】区块

    :param captions: {url: 图注}，同时充当白名单——只有参考内容里真实出现过的 URL 才放行
    :return: (去掉图片区块的正文, [{"url", "caption"}])
    """
    function_name = sys._getframe().f_code.co_name
    if IMAGE_MARKER not in text:
        return text.strip(), []

    answer, _, block = text.partition(IMAGE_MARKER)
    seen, kept = set(), []
    for line in block.splitlines():
        url = line.strip().strip("<>").strip()
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        if url in captions:
            kept.append({"url": url, "caption": captions[url]})

    dropped = len(seen) - len(kept)
    if dropped:
        logger.warning(
            f"[{NODE_NAME}] [{function_name}] 答案里有 {dropped} 个图片链接不在参考内容中，已过滤"
        )
    return answer.strip(), kept


def _generate(session_id: str, messages: list, is_stream: bool) -> str:
    """
    调 LLM 生成答案，返回未经处理的原文

    流式模式下逐块推送 delta。**图片区块不推给前端**——它只是给节点解析用的，
    一旦读到标记就停止推送（但仍继续累积原文，否则解析不出链接）。
    """
    llm = get_llm_client()

    if not is_stream:
        resp = llm.invoke(messages)
        return (getattr(resp, "content", "") or "").strip()

    buf = ""
    pushed = 0          # 已推送给前端的字符数
    cut = None          # 图片区块的起始位置
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", "") or ""
        if not piece:
            continue
        buf += piece
        if cut is None:
            idx = buf.find(IMAGE_MARKER)
            if idx >= 0:
                cut = idx
        visible_end = cut if cut is not None else len(buf)
        if visible_end > pushed:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": buf[pushed:visible_end]})
            pushed = visible_end
    return buf.strip()


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 生成答案 (node_answer_output)

    :param state: 需包含 session_id；有 answer 则直接输出，否则用 reranked_docs 生成
    :return: {"answer": 最终答案文本}
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 节点处理开始")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    preset = (state.get("answer") or "").strip()
    docs = state.get("reranked_docs") or []
    streamed = False    # 是否已经通过 delta 推过内容

    try:
        if preset:
            # 前置节点（产品名反问 / 拒识）已经给出答复，不必再调模型
            final_text, images = preset, []
            logger.info(f"[{NODE_NAME}] [{function_name}] 前置节点已产出答案，跳过 LLM 生成")
        elif not docs:
            final_text, images = FALLBACK_ANSWER, []
            logger.warning(f"[{NODE_NAME}] [{function_name}] 没有可用参考切片，返回兜底答复")
        else:
            context, captions = _build_context(docs)
            prompt = load_prompt(
                "answer_out",
                context=context,
                history=_build_history(session_id),
                item_names="、".join(state.get("item_names") or []) or "（无）",
                question=state.get("rewritten_query") or state.get("original_query") or "",
            )
            messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
            logger.info(
                f"[{NODE_NAME}] [{function_name}] 参考切片 {len(docs)} 条，"
                f"上下文 {len(context)} 字符，开始生成"
            )
            raw = _generate(session_id, messages, is_stream)
            streamed = is_stream
            final_text, images = _split_images(raw, captions)
            logger.info(
                f"[{NODE_NAME}] [{function_name}] 生成完成，答案 {len(final_text)} 字符，"
                f"配图 {len(images)} 张"
            )

        if is_stream:
            # 非 LLM 路径没有逐块推送，这里整段补一次 delta
            if not streamed:
                push_to_session(session_id, SSEEvent.DELTA, {"delta": final_text})
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": final_text,
                    "status": "completed",
                    "images": images,
                },
            )

        # 存档助手这一轮的答案，供 /history 接口读取
        try:
            save_chat_message(session_id, "assistant", final_text)
            logger.info(f"[{NODE_NAME}] [{function_name}] 助手消息已存档")
        except Exception as e:
            # 存档失败不应影响答案返回
            logger.error(f"[{NODE_NAME}] [{function_name}] 助手消息存档失败：{e}", exc_info=True)

        return {"answer": final_text, "images": images}

    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 生成失败：{e}", exc_info=True)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": f"答案生成失败：{e}"})
        return {"answer": ""}
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 节点处理结束")


if __name__ == '__main__':
    """
    本地测试：走真实检索 → 真实生成

    前置：Milvus / Neo4j / MongoDB 均在运行
    """
    from app.query_process.agent.main_graph import query_app
    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    cases = [
        ("正常问答", "Brother HAK 180 烫金机怎么安装烫金膜盒？"),
        ("库中无此产品", "小米15怎么开机？"),
    ]

    for label, question in cases:
        session_id = f"answer_test_{label}"
        logger.info("=" * 70)
        logger.info(f"[测试] {label}：{question}")
        st = create_query_default_state(
            session_id=session_id, original_query=question, is_stream=False,
        )
        try:
            result = query_app.invoke(st)
            answer = (result.get("answer") or "").strip()
            docs = result.get("reranked_docs") or []
            logger.info(f"[测试] 参考切片 {len(docs)} 条")
            logger.info(f"[测试] 答案（{len(answer)} 字符）：\n{answer[:400]}")
            if not answer:
                logger.error("[测试] [FAIL] 答案为空")
            if label == "正常问答":
                if not docs:
                    logger.error("[测试] [FAIL] 正常问答没有检索到切片")
                if "【图片】" in answer:
                    logger.error("[测试] [FAIL] 图片区块没被拆掉，仍留在正文里")
                if "测试回答" in answer or "打字机" in answer:
                    logger.error("[测试] [FAIL] 仍是占位文本")
        except Exception as e:
            logger.error(f"[测试] [FAIL] 执行失败：{e}", exc_info=True)
        finally:
            clear_task(session_id)

    logger.info("=" * 70)
    logger.info("[测试] 全部用例执行完毕")
