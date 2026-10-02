"""
单次调用记账

一次问答要调 LLM + 三路嵌入/检索 + 重排 + 联网搜索，花了多少、谁花的，此前账上一片空白。
本模块给每一次外部模型调用记一条账：**谁（trace/tenant/session）、在哪（node）、调了什么
（kind/model）、用了多少（tokens/延迟）、花了多少（成本估算）**。

三块职责：

1. **归因** —— ContextVar 里存「这次请求是谁发起的」。为什么不用参数传递：本项目的调用点
   散在 16 个节点的 8 处，逐个传参既侵入又必漏（漏了还不报错，只是账本上少一笔）。
   LangGraph 的并发执行器提交节点任务前会 `copy_context()`，所以入口处设一次，
   四路并发检索（各自在不同线程）都能读到同一份归因。
2. **采集** —— LLM 走 LangChain callback（`TokenUsageCallback`，挂在 `get_llm_client`）；
   嵌入 / 重排 / 联网搜索不是 LangChain 调用，各自手工埋点。
3. **落库** —— `record()` 统一写 MongoDB `llm_usage`（append-only）+ 打一行日志。

**记账绝不能反过来弄坏主流程**：所有落库与解析都吞异常并降级为 warning。
一次账单写不进去，不该让用户拿不到答案。

**流式也要有用量**：`answer_output` 是流式的，而 langchain-openai 只在 base_url 为
OpenAI 默认地址时才默认开启 `stream_usage`。本项目指向 DashScope，必须显式打开，
否则最贵的那次调用（大上下文生成答案）会漏记——已在 `lm_utils` 里显式传入。
"""
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

from app.conf.pricing_config import estimate_cost
from app.core.logger import logger

# 节点名，与各节点里的 NODE_NAME 含义一致，仅用于日志前缀
NODE_NAME = "usage_tracker"

# 当前请求的归因上下文。默认 None 表示「不在任何请求里」（如命令行单跑某节点），
# 此时照记不误，只是 tenant_id / node 等字段为空。
_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar("usage_context", default=None)


def new_trace_id() -> str:
    """生成一次请求的 trace_id（一次问答 = 一个 trace，会话里的多轮各自独立）"""
    return uuid.uuid4().hex[:16]


class UsageAccumulator:
    """
    单次请求的累计用量

    放进上下文里的是**这个对象本身**（可变），而 ContextVar 被复制到子上下文时复制的是
    绑定而非对象，因此四路并发检索各自记的账会累加到同一个实例上，请求结束时能直接
    报出「本次问答一共花了多少」，不必回 Mongo 聚合。
    """

    def __init__(self):
        self.calls = 0
        self.failed = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0
        self.unpriced_calls = 0   # 不在计价表里的调用（如联网搜索），成本按 0 计但单独标出
        self.started_at = time.time()
        # 分节点 / 分类型的明细，供前端「本次消耗」展开看
        self.by_node = {}
        self.by_kind = {}
        self._lock = threading.Lock()

    def _bucket(self, store: Dict[str, Dict[str, Any]], key: str, record: Dict[str, Any]) -> None:
        item = store.setdefault(key, {"calls": 0, "tokens": 0, "cost": 0.0})
        item["calls"] += 1
        item["tokens"] += record.get("total_tokens") or 0
        item["cost"] += record.get("cost") or 0.0

    def add(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self.calls += 1
            if not record.get("ok"):
                self.failed += 1
            self.prompt_tokens += record.get("prompt_tokens") or 0
            self.completion_tokens += record.get("completion_tokens") or 0
            cost = record.get("cost")
            if cost is None:
                self.unpriced_calls += 1
            else:
                self.cost += cost

            self._bucket(self.by_node, record.get("node") or "—", record)
            self._bucket(self.by_kind, record.get("kind") or "—", record)

    def summary(self) -> Dict[str, Any]:
        """本次请求的汇总，供服务入口打日志 / 前端实时展示"""
        with self._lock:
            return {
                "calls": self.calls,
                "failed": self.failed,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "cost": round(self.cost, 6),
                "unpriced_calls": self.unpriced_calls,
                "elapsed_ms": int((time.time() - self.started_at) * 1000),
                # 按成本降序，前端直接顺序渲染
                "by_node": [
                    {"node": k, "calls": v["calls"], "tokens": v["tokens"], "cost": round(v["cost"], 6)}
                    for k, v in sorted(self.by_node.items(), key=lambda kv: kv[1]["cost"], reverse=True)
                ],
                "by_kind": [
                    {"kind": k, "calls": v["calls"], "tokens": v["tokens"], "cost": round(v["cost"], 6)}
                    for k, v in sorted(self.by_kind.items(), key=lambda kv: kv[1]["cost"], reverse=True)
                ],
            }

    def text(self) -> str:
        """一行可读的汇总，直接进日志"""
        s = self.summary()
        extra = f"，{s['unpriced_calls']} 次未计价" if s["unpriced_calls"] else ""
        failed = f"，失败 {s['failed']} 次" if s["failed"] else ""
        return (
            f"调用 {s['calls']} 次{failed}，tokens {s['prompt_tokens']}+{s['completion_tokens']}"
            f"，估算成本 {s['cost']:.6f} 元{extra}，耗时 {s['elapsed_ms']}ms"
        )


def bind_context(**kwargs) -> Any:
    """
    在当前上下文里设置归因字段，返回 token 供 `reset_context` 还原

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
        # token 与当前上下文不匹配（跨上下文 reset）时忽略：记账上下文宁可脏一点，也不要抛
        pass


def current_context() -> Dict[str, Any]:
    """取当前归因上下文（去掉累计器与回调，避免调用方误改）"""
    return {k: v for k, v in (_CTX.get() or {}).items() if k not in ("acc", "on_usage")}


@contextmanager
def usage_context(trace_id: str = None, tenant_id: str = None, session_id: str = None,
                  node: str = None, on_usage=None, **extra):
    """
    一次请求的记账范围，用法：`with usage_context(session_id=..., tenant_id=...) as acc:`

    yield 出累计器，请求结束后用它 `.text()` 打一行总账。

    :param on_usage: 可选的进度回调，每记完一笔就带着**累计汇总**调一次。
        查询服务用它把用量实时推给前端（`usage` 事件），
        记账层因此不必知道 SSE 的存在——依赖方向保持单向。
        回调里的异常会被吞掉：前端显示不该拖垮问答。
    """
    acc = UsageAccumulator()
    token = bind_context(
        trace_id=trace_id or new_trace_id(),
        tenant_id=tenant_id,
        session_id=session_id,
        node=node,
        acc=acc,
        on_usage=on_usage,
        **extra,
    )
    try:
        yield acc
    finally:
        reset_context(token)


def tracked_node(name: str, fn):
    """
    包装图节点，执行期间把节点名写进归因上下文

    在 `main_graph` 注册节点处套一层即可，不必改动节点内部 —— 一个图的十几个节点
    只有一处需要维护，也不会漏。命令行单跑某节点时没有这层包装，账本里 node 为空。
    """

    @wraps(fn)
    def wrapper(state, *args, **kwargs):
        token = bind_context(node=name)
        try:
            return fn(state, *args, **kwargs)
        finally:
            reset_context(token)

    return wrapper


def add_tracked_node(graph, name: str, fn) -> None:
    """
    注册节点并顺带加上归因包装：`add_tracked_node(builder, "node_rerank", node_rerank)`

    两张图（导入 / 检索）注册节点时都用这个，省得每处都手写一层 tracked_node。
    """
    graph.add_node(name, tracked_node(name, fn))


def record(
    kind: str,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 0.0,
    ok: bool = True,
    error: str = "",
    **extra,
) -> Dict[str, Any]:
    """
    记一笔账：写日志 + 累加到当前请求 + 落 MongoDB

    :param kind: 调用类型，llm / embedding / rerank / mcp_search
    :param model: 模型名（不在计价表里时成本记为 None，而不是 0）
    :param latency_ms: 本次调用耗时
    :param ok: 调用是否成功；失败的调用同样入账，用于观察错误率
    :return: 落库的账目字典
    """
    ctx = _CTX.get() or {}
    cost = estimate_cost(model, prompt_tokens, completion_tokens) if ok else None

    item = {
        "ts": time.time(),
        "trace_id": ctx.get("trace_id") or "",
        "tenant_id": ctx.get("tenant_id") or "",
        "session_id": ctx.get("session_id") or "",
        "node": ctx.get("node") or "",
        "kind": kind,
        "model": model or "",
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        "latency_ms": int(latency_ms or 0),
        "cost": cost,
        "ok": ok,
        "error": (error or "")[:300],
    }
    if extra:
        item["extra"] = extra

    acc = ctx.get("acc")
    if acc is not None:
        acc.add(item)
        notify = ctx.get("on_usage")
        if notify is not None:
            # 进度回调失败只告警：前端显示不该影响账目，更不该影响问答
            try:
                notify(acc.summary())
            except Exception as e:
                logger.warning(f"[{NODE_NAME}] 用量回调失败（不影响业务）：{e}")

    logger.info(
        f"[记账] {kind:<10} model={model or '—':<18} "
        f"tokens={item['prompt_tokens']}+{item['completion_tokens']} "
        f"耗时={item['latency_ms']}ms "
        f"成本={'—' if cost is None else f'{cost:.6f}元'} "
        f"node={item['node'] or '—'} "
        f"{'' if ok else '失败 ' + item['error']}"
    )

    _persist(item)
    return item


def _persist(item: Dict[str, Any]) -> None:
    """落库。失败只告警——记账不该阻断业务"""
    try:
        from app.clients.mongo_usage_utils import save_usage
        save_usage(item)
    except Exception as e:
        logger.warning(f"[{NODE_NAME}] 账目落库失败（不影响业务）：{e}")


def _extract_usage(response) -> tuple:
    """
    从 LangChain 的 LLMResult 里挖出 token 用量

    两个来源，按可靠性优先：
    1. `usage_metadata` —— LangChain 标准字段，流式（stream_usage=True）与非流式都有
    2. `llm_output["token_usage"]` —— OpenAI 风格的旧字段，部分路径只给这个
    """
    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if not meta:
                info = getattr(generation, "generation_info", None) or {}
                meta = info.get("usage_metadata")
            if meta:
                return (
                    meta.get("input_tokens", 0) or 0,
                    meta.get("output_tokens", 0) or 0,
                )

    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage") or {}
    return (
        token_usage.get("prompt_tokens", 0) or 0,
        token_usage.get("completion_tokens", 0) or 0,
    )


class TokenUsageCallback(BaseCallbackHandler):
    """
    LLM 调用记账（挂在 `get_llm_client` 返回的客户端上，对全项目生效）

    计时靠 on_llm_start / on_chat_model_start 记开始时刻、on_llm_end 收尾；
    run_id 对并发调用唯一，四路检索并发时各算各的。
    """

    def __init__(self):
        super().__init__()
        self._starts: Dict[Any, tuple] = {}   # run_id -> (开始时刻, 模型名)
        self._lock = threading.Lock()

    # ----- 内部工具 -----

    def _begin(self, run_id, serialized):
        model = ""
        kwargs = (serialized or {}).get("kwargs") or {}
        model = kwargs.get("model_name") or kwargs.get("model") or ""
        with self._lock:
            self._starts[run_id] = (time.perf_counter(), model)

    def _finish(self, run_id):
        with self._lock:
            started, model = self._starts.pop(run_id, (None, ""))
        latency_ms = (time.perf_counter() - started) * 1000 if started else 0.0
        return model, latency_ms

    # ----- 回调入口 -----

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._begin(run_id, serialized)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._begin(run_id, serialized)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        try:
            model, latency_ms = self._finish(run_id)
            if not model:
                model = ((getattr(response, "llm_output", None) or {}).get("model_name") or "")
            prompt_tokens, completion_tokens = _extract_usage(response)
            record(
                "llm",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception as e:
            # 回调里抛异常会污染 LangChain 的调用链，这里必须兜住
            logger.warning(f"[{NODE_NAME}] LLM 记账失败（不影响调用）：{e}")

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        try:
            model, latency_ms = self._finish(run_id)
            record(
                "llm",
                model=model,
                latency_ms=latency_ms,
                ok=False,
                error=f"{type(error).__name__}: {error}",
            )
        except Exception as e:
            logger.warning(f"[{NODE_NAME}] LLM 失败记账失败：{e}")


class Timer:
    """计时小工具：`with Timer() as t: ...` 之后读 `t.ms`"""

    def __init__(self):
        self.ms = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1000
        return False


if __name__ == '__main__':
    """
    自测：验证归因传播、累加、落库

    不依赖外部服务也能验证前两项；MongoDB 不在线时落库只告警。
    **自测产生的账目会自己清掉**（打 selftest 标记），不污染真实账本。
    """
    from app.core.logger import logger as log

    problems = []
    MARK = {"selftest": True}

    # 1. 无上下文时记账不应报错（命令行单跑节点的场景）
    try:
        rec = record("llm", model="qwen-plus", prompt_tokens=100, completion_tokens=20,
                     latency_ms=123, **MARK)
        if rec["cost"] is None:
            problems.append("qwen-plus 应能算出成本")
        if rec["node"] or rec["tenant_id"]:
            problems.append("无上下文时 node / tenant 应为空")
    except Exception as e:
        problems.append(f"无上下文记账抛异常：{e}")

    # 2. 上下文内的归因 + 累加 + 实时回调
    seen = []
    with usage_context(session_id="usage_test", tenant_id="t_demo",
                       on_usage=lambda s: seen.append(s)) as acc:
        token = bind_context(node="node_rerank")
        record("rerank", model="gte-rerank-v2", prompt_tokens=1000, latency_ms=50, **MARK)
        reset_context(token)
        record("mcp_search", model="EnhancedSearch", latency_ms=800, **MARK)
        ctx = current_context()
        if ctx.get("session_id") != "usage_test":
            problems.append("session_id 未传播")
        if ctx.get("node"):
            problems.append("node 在 reset 后应被清掉")
        if "on_usage" in ctx:
            problems.append("on_usage 回调不应出现在 current_context 里")
        summ = acc.summary()
        if summ["calls"] != 2:
            problems.append(f"累计次数应为 2，实际 {summ['calls']}")
        if summ["unpriced_calls"] != 1:
            problems.append("未计价调用数应为 1（联网搜索没在计价表里）")
        nodes = {n["node"] for n in summ["by_node"]}
        if nodes != {"node_rerank", "—"}:
            problems.append(f"按节点归集不对：{nodes}")
        # 每记一笔推一次，且推的是累计值（不是单笔）
        if [s["calls"] for s in seen] != [1, 2]:
            problems.append(f"用量回调次数不对：{[s['calls'] for s in seen]}")
        if seen and seen[-1]["cost"] != summ["cost"]:
            problems.append("回调里的成本与汇总不一致")
        log.info(f"[测试] 本次请求汇总：{acc.text()}")

    # 3. 图节点的包装器
    def fake_node(state):
        return {"ok": current_context().get("node") == "wrapped"}

    wrapped = tracked_node("wrapped", fake_node)
    if not wrapped({}).get("ok"):
        problems.append("tracked_node 未把节点名写进上下文")
    if current_context():
        problems.append("tracked_node 退出后未还原上下文")

    # 4. 清掉本次自测写进账本的记录
    try:
        from app.clients.mongo_usage_utils import get_usage_tool
        removed = get_usage_tool().collection.delete_many({"extra.selftest": True}).deleted_count
        log.info(f"[测试] 已清理自测账目 {removed} 条")
    except Exception as e:
        log.warning(f"[测试] 自测账目清理失败（MongoDB 可能未启动）：{e}")

    if problems:
        for p in problems:
            log.error(f"[测试] [FAIL] {p}")
        sys.exit(1)
    log.success("[测试] [PASS] 记账核心模块验证通过")
