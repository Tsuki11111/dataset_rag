"""
知识库查询 Web 服务

把 LangGraph 检索流程（产品名确认 → 四路检索 → RRF 融合 → 重排 → 生成答案）
包装成 HTTP 接口，并用 SSE 实时推送执行流程与流式答案。

四个核心模块协同：
1. Web 服务层（本文件）：接收请求、建立 SSE 连接
2. SSE 工具层（app/utils/sse_utils.py）：消息队列、打包、推送事件
3. 任务状态层（app/utils/task_utils.py）：记录节点执行进度，并触发 SSE 推送
4. 图节点执行层（app/query_process/agent/）：业务节点，更新状态驱动进度

时序：
    POST /query（is_stream=true） → 建 SSE 队列 → 后台跑图 → 立即返回 session_id
    GET  /stream/{session_id}    → 前端订阅，持续收到 progress / delta / final
"""
import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.clients.mongo_history_utils import clear_history, get_recent_messages
from app.core.logger import logger
from app.core.usage_tracker import current_context, usage_context
from app.query_process.agent.main_graph import query_app
from app.utils.auth_utils import clear_session_cookie, current_tenant, set_session_cookie
from app.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    set_task_result,
    update_task_status,
)

# 服务端口：8000 被 Attu 占用，8001 被导入服务使用，故查询服务用 8002
SERVICE_PORT = 8002

# 节点名，用于日志前缀
NODE_NAME = "query_service"

app = FastAPI(
    title="Query Service",
    description="掌柜智库知识库查询服务（SSE 流式推送执行流程与答案）"
)

# 跨域配置：允许前端页面独立部署时调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="用户问题")
    session_id: str = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


def run_query_graph(session_id: str, user_query: str, is_stream: bool = True,
                    tenant_id: str = None) -> dict:
    """
    后台执行检索图

    由 BackgroundTasks 触发，不阻塞 HTTP 响应。图内各节点会自行更新任务进度，
    而 task_utils 的进度更新会通过 push_to_session 推给 SSE 连接（仅流式模式）。

    整个执行过程包在 `usage_context` 里：一次问答 = 一个 trace，图内所有模型调用
    （含四路并发检索，各自在不同线程）都归到这条 trace 与调用方租户上。

    :param session_id: 会话ID，同时作为 SSE 队列的 key
    :param user_query: 用户原始问题
    :param is_stream: 是否流式推送
    :param tenant_id: 调用方租户，来自访问密钥
    :return: 本次问答的用量汇总（调用次数 / tokens / 估算成本）
    """
    function_name = sys._getframe().f_code.co_name

    init_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream,
    }
    # 流式模式下把用量实时推给前端（「本次消耗」那一行）：
    # 每记完一笔调一次，前端边看答案边看钱。
    # 非流式没有 SSE 队列，不注册回调，避免 push_to_session 刷「没有队列」的告警。
    on_usage = None
    if is_stream:
        on_usage = lambda summary: push_to_session(session_id, SSEEvent.USAGE, summary)

    with usage_context(session_id=session_id, tenant_id=tenant_id, node="query_graph",
                       on_usage=on_usage) as acc:
        logger.info(
            f"[{NODE_NAME}] [{function_name}] 开始执行检索图，"
            f"session={session_id}，trace={current_context().get('trace_id')}"
        )
        try:
            final_state = query_app.invoke(init_state)

            # 把最终答案存入任务结果，供非流式模式取用
            answer = (final_state or {}).get("answer", "")
            set_task_result(session_id, "answer", answer)
            # 配图同样要带出去，否则非流式模式拿不到
            set_task_result(session_id, "images", (final_state or {}).get("images") or [])
            # push_queue=is_stream：只有流式模式才推送进度，避免无连接时产生告警噪音
            update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
            logger.info(f"[{NODE_NAME}] [{function_name}] 检索图执行完成，session={session_id}")
        except Exception as e:
            logger.error(f"[{NODE_NAME}] [{function_name}] 检索图执行失败：{e}", exc_info=True)
            update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
            if is_stream:
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

    # 记账汇总放在 with 之外：退出上下文只是清掉归因，累计器还能读
    summary = acc.summary()
    logger.info(f"[{NODE_NAME}] [{function_name}] 本次问答记账：{acc.text()}")
    return summary


class LoginRequest(BaseModel):
    """登录请求：提交访问密钥"""
    key: str


@app.on_event("startup")
async def on_startup():
    """
    服务启动时预热用户表

    顺带检查有没有用户——一个都没有时所有数据接口都会 401，提前把话说明白。
    """
    from app.clients.mongo_user_utils import count_users, get_user_tool
    get_user_tool()
    logger.info("用户工具已就绪（MongoDB）")
    if count_users() == 0:
        logger.warning(
            "还没有任何用户，数据接口将全部返回 401。"
            "先建一个：.venv/Scripts/python.exe -m app.clients.mongo_user_utils add <称呼>"
        )


@app.post("/login", summary="登录：校验访问密钥并写入会话 Cookie")
async def login(payload: LoginRequest, response: Response):
    """
    用访问密钥换一个 HttpOnly Cookie

    走 Cookie 而不是让前端存密钥发请求头：本服务的流式接口用 `EventSource`，
    它**无法自定义请求头**，只有 Cookie 能被浏览器自动携带。
    """
    function_name = sys._getframe().f_code.co_name
    from app.clients.mongo_user_utils import verify_key

    user = verify_key(payload.key)
    if not user:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 登录失败：密钥无效或已撤销")
        raise HTTPException(status_code=401, detail="访问密钥无效或已撤销")

    set_session_cookie(response, payload.key)
    logger.info(
        f"[{NODE_NAME}] [{function_name}] 登录成功：{user['name']}"
        f"（{user['role']}，租户 {user['tenant_id']}）"
    )
    return {"code": 200, "name": user["name"], "role": user["role"]}


@app.post("/logout", summary="退出登录")
async def logout(response: Response):
    """清掉会话 Cookie"""
    clear_session_cookie(response)
    return {"code": 200}


@app.get("/chat.html", summary="聊天页面")
async def chat():
    """返回前端聊天页面"""
    # 本文件位于 app/query_process/api/，页面在 app/query_process/page/
    page_path = Path(__file__).absolute().parent.parent / "page" / "chat.html"
    if not page_path.exists():
        logger.error(f"聊天页面不存在：{page_path}")
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{page_path}！")
    return FileResponse(page_path, media_type="text/html")


@app.post("/query", summary="提交查询")
async def query(background_tasks: BackgroundTasks, request: QueryRequest,
                user: dict = Depends(current_tenant)):
    """
    接收用户提问并启动后台检索流程

    流式模式：建 SSE 队列 → 后台跑图 → 立即返回 session_id（前端随即订阅 /stream）
    非流式模式：同步跑完图后直接返回答案

    鉴权用参数形式而非 `dependencies=[...]`：需要拿到 tenant_id 才能把这次问答的
    成本记到调用方名下。
    """
    function_name = sys._getframe().f_code.co_name
    user_query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    tenant_id = user.get("tenant_id")

    logger.info(f"[{NODE_NAME}] [{function_name}] 收到查询，session={session_id}，流式={is_stream}，问题={user_query}")

    if is_stream:
        # 建队列必须在跑图之前：否则节点推送时队列还不存在，事件会丢失
        create_sse_queue(session_id)
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream, tenant_id)
        return {
            "message": "结果正在处理中...",
            "session_id": session_id,
        }

    # 非流式：同步执行，直接返回答案
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    usage = run_query_graph(session_id, user_query, is_stream, tenant_id)
    answer = get_task_result(session_id, "answer", "")
    images = get_task_result(session_id, "images", [])
    done_list = get_done_task_list(session_id)
    clear_task(session_id)
    return {
        "message": "处理完成！",
        "session_id": session_id,
        "answer": answer,
        "images": images,
        "done_list": done_list,
        # 本次问答花了多少：调用次数 / tokens / 估算成本（明细见 llm_usage 集合）
        "usage": usage,
    }


@app.get("/stream/{session_id}", summary="SSE 流式获取结果", dependencies=[Depends(current_tenant)])
async def stream(session_id: str, request: Request):
    """
    建立 SSE 长连接，实时推送任务进度与生成文本

    推送的事件类型：
    - ready    ：连接建立
    - progress ：节点进度（status / done_list / running_list）
    - delta    ：答案的增量字符（打字机效果）
    - final    ：完整答案与状态
    - error    ：执行异常
    """
    logger.info(f"[{NODE_NAME}] [stream] 建立SSE连接，session={session_id}")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 禁用Nginx等反向代理的缓冲，否则事件会被攒着一起发
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health", summary="健康检查")
async def health():
    """检查服务是否正常"""
    return {"ok": True}


@app.get("/history/{session_id}", summary="查询会话历史", dependencies=[Depends(current_tenant)])
async def get_history(session_id: str, limit: int = 50):
    """
    查询指定会话的历史对话记录（时间正序）

    :param session_id: 会话ID
    :param limit: 返回条数上限，默认50
    """
    function_name = sys._getframe().f_code.co_name
    try:
        rows = get_recent_messages(session_id, limit=limit)
    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 查询历史失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询历史失败：{e}")

    items = [{
        "_id": str(r.get("_id")) if r.get("_id") is not None else "",
        "session_id": r.get("session_id", ""),
        "role": r.get("role", ""),
        "text": r.get("text", ""),
        "rewritten_query": r.get("rewritten_query", ""),
        "item_names": r.get("item_names") or [],
        "ts": r.get("ts"),
    } for r in rows]

    logger.info(f"[{NODE_NAME}] [{function_name}] 会话{session_id}历史查询，返回{len(items)}条")
    return {"session_id": session_id, "items": items}


@app.delete("/history/{session_id}", summary="清空会话历史", dependencies=[Depends(current_tenant)])
async def clear_session_history(session_id: str):
    """删除指定会话的全部历史对话记录"""
    function_name = sys._getframe().f_code.co_name
    try:
        deleted = clear_history(session_id)
    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 清空历史失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清空历史失败：{e}")

    logger.info(f"[{NODE_NAME}] [{function_name}] 会话{session_id}历史已清空，删除{deleted}条")
    return {"message": "History cleared", "session_id": session_id, "deleted_count": deleted}


if __name__ == "__main__":
    logger.info(f"[{NODE_NAME}] 查询服务启动中（端口{SERVICE_PORT}）...")
    uvicorn.run(app=app, host="127.0.0.1", port=SERVICE_PORT)
