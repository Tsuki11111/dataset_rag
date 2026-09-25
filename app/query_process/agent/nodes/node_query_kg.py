import sys
import time

from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_query_kg"


def node_query_kg(state):
    """
    节点: 图谱查询 (node_query_kg)
    为什么叫这个名字: 在 Neo4j 知识图谱中查询实体关系 (Knowledge Graph)。
    节点功能：在 Neo4j 知识图谱中查询与用户问题相关的实体关系，补充结构化检索结果。
    输入：state['rewritten_query'] / state['item_names']
    输出：更新 state['kg_chunks']

    未来要实现:
    1. 连接 Neo4j。
    2. 根据产品名与问题抽取实体，执行 Cypher 查询。
    3. 将命中的实体关系切片写入 state['kg_chunks']。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 图谱查询处理开始")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    # 骨架阶段：仅占位，后续接入 Neo4j 查询
    time.sleep(1)

    add_done_task(state["session_id"], function_name, state.get("is_stream"))
    logger.info(f"[{NODE_NAME}] [{function_name}] 图谱查询处理结束")
    return {"kg_chunks": []}
