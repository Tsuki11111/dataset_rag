import sys
import time

from app.clients.mongo_history_utils import save_chat_message
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_item_name_confirm"


def node_item_name_confirm(state):
    """
    节点: 确认问题产品 (node_item_name_confirm)
    节点功能：确认用户问题中的核心产品名称。
    输入：state['original_query']
    输出：更新 state['item_names'] / state['rewritten_query']

    未来要实现:
    1. 结合历史对话提取产品名，把模糊问题改写为完整独立的精准问题。
    2. 将提取出的产品名在 Milvus 向量库中做检索，按评分对齐标准型号。
    3. 无法唯一确定时生成反问句（多选一 / 查无此人）写入 state['answer']，
       触发主图的条件边直接跳到答案输出，跳过后续检索。
    4. 把用户问题、改写后的问题、确认的产品名写入 MongoDB 历史记录。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    # 骨架阶段：后续接入大模型做产品名提取与问题改写
    time.sleep(1)

    add_done_task(state["session_id"], function_name, state.get("is_stream"))
    logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")

    item_names = state.get("item_names") or ["示例产品"]
    # 存档用户这一轮的问题（含改写结果与识别出的产品名），供 /history 接口读取
    try:
        save_chat_message(
            state["session_id"], "user", state["original_query"],
            state.get("rewritten_query", ""), item_names,
        )
        logger.info(f"[{NODE_NAME}] [{function_name}] 用户消息已存档")
    except Exception as e:
        # 存档失败不应中断检索主流程
        logger.error(f"[{NODE_NAME}] [{function_name}] 用户消息存档失败：{e}", exc_info=True)

    return {"item_names": item_names}
