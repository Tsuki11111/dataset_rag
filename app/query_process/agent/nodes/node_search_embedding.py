import time
import sys
from app.utils.task_utils import  add_done_task,add_running_task

def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    print("---量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    print("量内容检索答案！！")
    time.sleep(1)

    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---量内容检索 处理结束---")
    return {"embedding_chunks": []}