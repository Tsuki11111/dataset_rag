"""
请求上下文（日志与记账共用的归因）

一次请求里「这是谁发起的、正在哪个节点跑」这些信息，**日志要用、记账也要用**。
所以把存它的 ContextVar 单独放进这个**不导入任何项目模块**的文件里：
`logger.py` 要读它、`usage_tracker.py` 也要读它，而 `usage_tracker.py` 又依赖
`logger.py`——两边都放一起就会成环。这里做叶子模块是为了解开这个环。

**为什么用 ContextVar 而不是函数参数**：调用点在 16 个节点里散落着，
逐个传参既侵入又必漏，而**漏了不报错**——只是日志上少两列、账本上少一笔。
设一次即可全局生效，靠的是 LangGraph 的并发执行器提交节点任务前会
`copy_context()`，所以四路并发检索（各自在不同线程）都能读到同一份归因。

字段约定（都可以为空，比如命令行单跑某个节点时）：

| 字段 | 含义 | 谁设置 |
|---|---|---|
| `trace_id` | 一次请求的唯一定位符。一次问答 / 一次导入各一个 | `usage_context` |
| `tenant_id` | 调用方租户，来自访问密钥 | 服务入口 |
| `session_id` | 会话 ID（问答）或任务 ID（导入） | 服务入口 |
| `node` | 当前正在执行的图节点 | `add_tracked_node` 包装器 |

`usage_tracker` 还会往里塞两个内部键（`acc` 累计器、`on_usage` 回调），
`current_context()` 会把它们过滤掉，不对外暴露。
"""
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# 内部键：累计器与回调，只给记账层用，不对外暴露
INTERNAL_KEYS = ("acc", "on_usage")

_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar("request_context", default=None)

# 日志里展示的字段（不含内部键）
PUBLIC_FIELDS = ("trace_id", "tenant_id", "session_id", "node")


def new_trace_id() -> str:
    """生成一次请求的 trace_id（一次问答 = 一个 trace，会话里的多轮各自独立）"""
    return uuid.uuid4().hex[:16]


def bind_context(**kwargs) -> Any:
    """
    在当前上下文里设置字段，返回 token 供 `reset_context` 还原

    合并语义：只覆盖传入的键，其余保留（如节点包装器只设 node，不动 trace/tenant）。
    """
    base = dict(_CTX.get() or {})
    base.update(kwargs)
    return _CTX.set(base)


def reset_context(token: Any) -> None:
    """还原到 `bind_context` 之前的上下文"""
    try:
        _CTX.reset(token)
    except (ValueError, LookupError):
        # token 与当前上下文不匹配（跨上下文 reset）时忽略：
        # 上下文脏一点可以接受，为此抛异常会连累正在跑的请求
        pass


def current_context() -> Dict[str, Any]:
    """取当前归因上下文（已去掉累计器与回调等内部键）"""
    return {k: v for k, v in (_CTX.get() or {}).items() if k not in INTERNAL_KEYS}


def raw_context() -> Dict[str, Any]:
    """
    取**未过滤**的上下文，含 `acc` / `on_usage` 等内部键

    只给记账层用（它要把每笔账累加到 `acc` 上、并回调 `on_usage`）。
    其他模块一律用 `current_context()`。
    """
    return dict(_CTX.get() or {})


def context_fields() -> Dict[str, str]:
    """
    取用于日志的结构化字段，值统一成字符串（便于 JSON 序列化与对齐）

    缺省为空字符串而不是 None：日志里 `"trace_id": ""` 比 `null` 更好处理，
    jq 里 `select(.trace_id != "")` 直接可用。
    """
    ctx = _CTX.get() or {}
    return {k: str(ctx.get(k) or "") for k in PUBLIC_FIELDS}


if __name__ == '__main__':
    """自测：设置、合并、还原、隔离"""
    from app.core.logger import logger

    problems = []

    if current_context():
        problems.append("初始状态应为空")

    token = bind_context(trace_id="t1", tenant_id="tenant_a", node="node_x")
    if context_fields()["trace_id"] != "t1":
        problems.append("trace_id 未写入")

    # 合并语义：只改 node，别的字段要留着
    inner = bind_context(node="node_y")
    if context_fields()["node"] != "node_y" or context_fields()["tenant_id"] != "tenant_a":
        problems.append("合并语义不对：应只覆盖传入的键")
    reset_context(inner)
    if context_fields()["node"] != "node_x":
        problems.append("reset 后应回到 node_x")

    # 子上下文（模拟 LangGraph 的 copy_context）里改动不影响父上下文
    import contextvars
    def in_child():
        t = bind_context(node="node_child")
        seen = context_fields()["node"]
        reset_context(t)
        return seen
    child_seen = contextvars.copy_context().run(in_child)
    if child_seen != "node_child" or context_fields()["node"] != "node_x":
        problems.append("子上下文隔离不对")

    # 内部键不外泄
    bind_context(acc=object(), on_usage=lambda s: None)
    if set(current_context()) & set(INTERNAL_KEYS):
        problems.append("current_context 泄漏了内部键")

    reset_context(token)

    for p in problems:
        logger.error(f"[测试] [FAIL] {p}")
    if not problems:
        logger.success("[测试] [PASS] 请求上下文验证通过")
