"""
百炼 MCP 联网搜索客户端（EnhancedSearch 服务）

用 openai-agents 的 MCPServerStreamableHttp 客户端连接百炼 MCP。

与教程的差异（重要）：
1. **传输类不同**。教程用 MCPServerSse 连 `/sse` 端点，但本服务实测 `/sse` 返回 200
   却一个字节都不推（挂起 25 秒无输出），MCPServerSse 依赖服务端先发 `endpoint`
   事件，因此用不了。改用 MCPServerStreamableHttp 连 `/mcp` 端点。
2. **mcp 版本必须钉在 1.x**。mcp 2.x 改用 `server/discover` 新握手（协议 2026-07-28），
   百炼服务端仍是 2024-11-05 老协议，收到会直接返回 HTTP 500。
   pyproject 里已加 `mcp<2` 约束，不要升级。
3. **工具名与参数**。本服务只有 search_pro，且**只接受 query**，
   照教程多传 count 会直接 isError。
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger
from app.core.usage_tracker import Timer, record

# 百炼增强搜索的工具名
TOOL_NAME = "search_pro"
# 连接与流式读取超时（秒）
TIMEOUT = 60
SSE_READ_TIMEOUT = 300
# 工具一次返回约 10 条，按教程意图只取前 N 条交给下游重排
MAX_RESULTS = 5


class McpSearchError(Exception):
    """MCP 调用失败：配置缺失或结果解析异常"""


async def mcp_call(query: str):
    """
    异步调用百炼 MCP 搜索服务

    :param query: 搜索查询词（通常是改写后的问题）
    :return: 原始 CallToolResult；调用失败返回 None
    """
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": "Bearer " + mcp_config.api_key},
            "timeout": TIMEOUT,
            "sse_read_timeout": SSE_READ_TIMEOUT,
        },
    )

    try:
        logger.info(f"[MCP] 正在连接百炼搜索服务：{mcp_config.mcp_base_url}")
        await search_mcp.connect()

        logger.info(f"[MCP] 连接成功，调用工具 {TOOL_NAME} 查询：{query}")
        result = await search_mcp.call_tool(
            tool_name=TOOL_NAME,
            # 该工具只接受 query，传 count 会直接报 isError
            arguments={"query": query},
        )
        logger.info("[MCP] 工具调用完成")
        return result
    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常：{e}", exc_info=True)
        return None
    finally:
        # 无论成功/失败都关闭连接，避免资源泄漏
        await search_mcp.cleanup()


def _parse_docs(result) -> list[dict]:
    """把 MCP 原始返回值清洗为 [{title, url, snippet}]，截断到 MAX_RESULTS"""
    if result is None:
        logger.warning("[MCP] 返回结果为空或无效")
        return []
    if result.isError:
        logger.error(f"[MCP] 返回错误：{result}")
        return []
    if not result.content:
        logger.warning("[MCP] 返回内容为空")
        return []

    raw_text = result.content[0].text
    try:
        pages = (json.loads(raw_text) or {}).get("pages") or []
    except json.JSONDecodeError:
        logger.error(f"[MCP] 结果解析 JSON 失败：{raw_text[:100]}...")
        return []

    logger.info(f"[MCP] 原始页面数量：{len(pages)}")
    docs = []
    for item in pages:
        snippet = (item.get("snippet") or "").strip()
        if not snippet:
            # 没有摘要的结果对下游重排没有价值
            continue
        docs.append({
            "title": (item.get("title") or "").strip(),
            "url": (item.get("url") or "").strip(),
            "snippet": snippet,
        })
        if len(docs) >= MAX_RESULTS:
            break
    return docs


def _run_coro(coro):
    """
    在同步节点里跑异步协程（同步-异步桥接）

    没有运行中的事件循环时（CLI 测试、BackgroundTasks 线程池）直接用 asyncio.run。
    但非流式 HTTP 路径是在 async 路由里直接调图的，此时事件循环已在运行，
    asyncio.run 会抛 RuntimeError，故另开线程执行。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def search_web(query: str) -> list[dict]:
    """
    同步入口：调用百炼 MCP 搜索并返回结构化结果

    :param query: 搜索查询词
    :return: [{"title": ..., "url": ..., "snippet": ...}]，最多 MAX_RESULTS 条
    :raises McpSearchError: 配置缺失
    """
    if not mcp_config.mcp_base_url:
        raise McpSearchError("MCP_DASHSCOPE_BASE_URL 未配置")
    if not mcp_config.api_key:
        raise McpSearchError("OPENAI_API_KEY 未配置")

    timer = Timer()
    try:
        with timer:
            docs = _parse_docs(_run_coro(mcp_call(query)))
    except Exception as e:
        record("mcp_search", model=TOOL_NAME, latency_ms=timer.ms, ok=False, error=str(e))
        raise

    # 非 LangChain 调用，手工埋点。百炼增强搜索按次计费、单价未公开，
    # 故不传 tokens —— 记账里成本记为 None 并单独计数，而不是假装它免费
    record("mcp_search", model=TOOL_NAME, latency_ms=timer.ms, results=len(docs))
    return docs


if __name__ == '__main__':
    """
    本地测试：验证 MCP 端点连通性与结果清洗

    前置：.env 已配置 MCP_DASHSCOPE_BASE_URL 与 OPENAI_API_KEY
    """
    logger.info("=" * 70)
    logger.info("[测试] 开始验证百炼 MCP 联网搜索客户端")

    try:
        docs = search_web("烫金机的日常保养注意事项")
        logger.info(f"[测试] 返回 {len(docs)} 条（上限 {MAX_RESULTS}）")
        for i, d in enumerate(docs, 1):
            logger.info(f"[测试] {i}. {d['title'][:38]!r}")
            logger.info(f"[测试]     {d['url'][:70]}")
            logger.info(f"[测试]     摘要 {len(d['snippet'])} 字符：{d['snippet'][:60]!r}")

        problems = []
        if not docs:
            problems.append("没有返回任何结果")
        if len(docs) > MAX_RESULTS:
            problems.append(f"超出上限：{len(docs)} > {MAX_RESULTS}")
        if any(not d["snippet"] for d in docs):
            problems.append("存在 snippet 为空的条目（应被过滤）")
        if any(not d["url"] for d in docs):
            problems.append("存在 url 为空的条目")

        if problems:
            for p in problems:
                logger.error(f"[测试] [FAIL] {p}")
        else:
            logger.success("[测试] [PASS] 客户端验证通过")
    except Exception as e:
        logger.error(f"[测试] [FAIL] 调用失败：{e}", exc_info=True)

    logger.info("=" * 70)
