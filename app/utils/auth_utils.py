"""
访问鉴权（FastAPI 依赖）

提供 `current_tenant`：校验调用方身份，返回 `{tenant_id, name, role}`。

**两种密钥来源**：

- `Authorization: Bearer <key>` —— 给脚本 / 程序化调用
- Cookie `kb_session` —— 给浏览器

为什么必须支持 Cookie：查询服务的流式接口用的是 `EventSource`，
而 **EventSource 无法自定义请求头**，Bearer 头在那条路上根本发不出去。
Cookie 由浏览器自动携带，页面里十来处调用一行都不用改。

**目前只做「认证」、不做「授权过滤」**：`tenant_id` 已经拿到，但数据还没按它隔离
（给四个存储加租户维度是停机迁移级别的改动）。所以现阶段**任何有效密钥都能看到全部数据**——
这一层的价值是把身份打通，并为后续隔离留好接口，不是现在就实现隔离。
"""
import os
from typing import Any, Dict, Optional

from fastapi import Cookie, Header, HTTPException, Response

from app.clients.mongo_user_utils import verify_key

# 浏览器侧会话 Cookie 名
COOKIE_NAME = "kb_session"
# Cookie 有效期（秒）：30 天，够用且不至于永久
COOKIE_MAX_AGE = 30 * 24 * 3600


def _extract_key(authorization: Optional[str], kb_session: Optional[str]) -> str:
    """从请求头或 Cookie 里取出明文密钥，优先请求头"""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return (kb_session or "").strip()


def current_tenant(
    authorization: Optional[str] = Header(None),
    kb_session: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """
    校验调用方身份，作为路由依赖使用

    用法：`@app.get("/xxx", dependencies=[Depends(current_tenant)])`；
    需要用到身份时把 `user: Dict = Depends(current_tenant)` 写进参数。

    :raises HTTPException: 401 —— 未带密钥、密钥无效或已撤销
    """
    raw_key = _extract_key(authorization, kb_session)
    if not raw_key:
        raise HTTPException(status_code=401, detail="缺少访问密钥")

    user = verify_key(raw_key)
    if not user:
        raise HTTPException(status_code=401, detail="访问密钥无效或已撤销")
    return user


def set_session_cookie(response: Response, raw_key: str) -> None:
    """
    登录成功时把密钥写进 HttpOnly Cookie

    HttpOnly 让页面 JS 读不到，XSS 也偷不走；SameSite=Lax 挡掉跨站发起的写操作。
    """
    response.set_cookie(
        key=COOKIE_NAME,
        value=raw_key,
        httponly=True,
        samesite="lax",
        # 本机跑的是 http，置 True 会导致 Cookie 根本存不下来；
        # 部署到 https 后务必把 .env 的 COOKIE_SECURE 设为 1
        secure=os.getenv("COOKIE_SECURE") == "1",
        max_age=COOKIE_MAX_AGE,
    )


def clear_session_cookie(response: Response) -> None:
    """退出登录：清掉会话 Cookie"""
    response.delete_cookie(COOKIE_NAME)
