import time
import sys
from app.clients.mongo_history_utils import save_chat_message
from app.core.logger import logger
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_answer_output"


def node_answer_output(state):
    """
    节点: 生成答案 (node_answer_output)
    节点功能：进行过处理可以是流式输出可以整体输出！
    职责：
    1. 若前置节点（如产品名确认）已产出 answer（反问/拒绝），直接输出，跳过 LLM 生成。
    2. 否则组装 Prompt 调用大模型生成答案。
    3. 流式模式下逐字推送 delta，结束后推送 final。
    4. 把助手答案存档到 MongoDB，供 /history 接口读取。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 节点处理开始")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    base_answer = state.get("answer") or f"这是关于「{state.get('original_query', '当前问题')}」的测试回答，正在演示打字机流式输出效果。"
    final_text = ""

    if is_stream:
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.03)

        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": final_text,
                "status": "completed",
                # 骨架阶段占位图；后续应从 reranked_docs 的切片里解析真实图片 URL
                "image_urls": ["https://example.com/demo-1.png", "https://example.com/demo-2.png"]
            }
        )
        logger.info(f"[{NODE_NAME}] [{function_name}] 流式输出完成，总长度: {len(final_text)}")
    else:
        final_text = base_answer

    # 存档助手这一轮的答案，供 /history 接口读取
    try:
        save_chat_message(session_id, "assistant", final_text)
        logger.info(f"[{NODE_NAME}] [{function_name}] 助手消息已存档")
    except Exception as e:
        # 存档失败不应影响答案返回
        logger.error(f"[{NODE_NAME}] [{function_name}] 助手消息存档失败：{e}", exc_info=True)

    add_done_task(state['session_id'], function_name, state.get("is_stream"))
    logger.info(f"[{NODE_NAME}] [{function_name}] 节点处理结束")
    return {"answer": final_text}