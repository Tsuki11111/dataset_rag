"""
联网搜索节点 (node_web_search_mcp)

作用：调百炼 MCP 增强搜索，为知识库检索补充外部信息 ——
适合问设备常见问题、产品线信息等库里没有覆盖的内容。

检索词取 rewritten_query（含产品名、指代已消解），为空时回退 original_query。

实现方式与教程一致：调用异步 MCP SDK，在同步节点里用 asyncio 桥接。
传输类与依赖版本的细节（为什么不是 MCPServerSse、为什么 mcp 钉在 1.x）
见 app/clients/mcp_search_utils.py 的模块说明。
"""
import sys

from app.clients.mcp_search_utils import McpSearchError, search_web
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_web_search_mcp"


def node_web_search_mcp(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 联网搜索 (node_web_search_mcp)

    流程：
    1. 取检索词（改写后的问题优先）
    2. 调百炼 MCP 增强搜索
    3. 返回 web_search_docs

    :param state: 需包含 session_id / rewritten_query
    :return: {"web_search_docs": [{"title","url","snippet"}]}；失败或无结果返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    try:
        query = state.get("rewritten_query") or state.get("original_query")
        if not query:
            logger.warning(f"[{NODE_NAME}] [{function_name}] 无有效检索词，跳过联网搜索")
            return {"web_search_docs": []}

        logger.info(f"[{NODE_NAME}] [{function_name}] 检索词：{query!r}")
        docs = search_web(query)
        logger.info(f"[{NODE_NAME}] [{function_name}] 检索完成，返回 {len(docs)} 条")
        if docs:
            logger.info(
                f"[{NODE_NAME}] [{function_name}] Top1：{docs[0]['title'][:40]!r}"
            )
        return {"web_search_docs": docs}

    except McpSearchError as e:
        # 联网是四路召回之一，失败不应中断整条链路：返回空结果，RRF 会忽略空路
        logger.error(f"[{NODE_NAME}] [{function_name}] 联网搜索失败：{e}")
        return {"web_search_docs": []}
    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 联网搜索异常：{e}", exc_info=True)
        return {"web_search_docs": []}
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")


if __name__ == '__main__':
    """
    本地测试：验证节点的取词、调用、返回结构

    前置：.env 已配置 MCP_DASHSCOPE_BASE_URL 与 OPENAI_API_KEY
    """
    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    cases = [
        ("正常检索", "HAK 180 烫金机怎么安装烫金膜盒？"),
        ("无检索词", ""),
    ]

    for label, query in cases:
        session_id = f"mcp_test_{label}"
        logger.info("=" * 70)
        logger.info(f"[测试] {label}：query={query!r}")
        st = create_query_default_state(
            session_id=session_id,
            original_query=query,
            rewritten_query=query,
            is_stream=False,
        )
        try:
            result = node_web_search_mcp(st)
            docs = result.get("web_search_docs") or []
            logger.info(f"[测试] 返回 {len(docs)} 条")
            if docs:
                logger.info(f"[测试] 首条：{docs[0]['title'][:40]!r} -> {docs[0]['url'][:60]}")
        except Exception as e:
            logger.error(f"[测试] 执行失败：{e}", exc_info=True)
        finally:
            clear_task(session_id)

    logger.info("=" * 70)
    logger.info("[测试] 全部用例执行完毕")
