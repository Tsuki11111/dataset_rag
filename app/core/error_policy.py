"""
异常分级策略

项目里有十几处「某一路失败就降级、链路继续」的位置（四路召回、图谱、联网、重排……）。
它们此前一律写成 `except Exception: 记日志 + 返回空`，**把编程错误和外部故障混为一谈**：

> 实现 `node_query_kg` 时一个 `NameError` 被「失败不中断链路」的兜底 `except` 降级成 warning，
> 图谱那一路静默返回空、功能等于废了，只有测试断言才发现。

所以这里先把异常**分类**，再按类决定处置方式：

| 分类 | 什么情况 | 怎么处置 |
|---|---|---|
| `FATAL` | 代码自身不一致：`NameError` / `UnboundLocalError` / `ImportError` / `NotImplementedError` / `AssertionError` / `SyntaxError` / `IndentationError` / `RecursionError` | **上抛**。这类问题不能降级掩盖——降级只会让功能静默失效 |
| `RETRYABLE` | 暂时性外部故障：超时、连接失败、429、5xx、`ServerSelectionTimeoutError` | 降级 + warning。**Phase 2 的重试接在这里** |
| `BLOCKED` | 重试无用且要人处理：4xx（鉴权/参数）、配置缺失 | 降级 + error（带堆栈） |
| `UNEXPECTED` | 兜底：外部依赖的其它异常、数据结构不符预期 | 降级 + error（带堆栈） |

**为什么 `TypeError` / `KeyError` / `AttributeError` / `IndexError` 不算 FATAL**：
它们既能由我们写错引起，也能由外部返回的数据变形引起（SDK 改结构、接口少字段），
在降级点上分不清。**归类为 `UNEXPECTED`**：仍然降级（保住"一路坏不影响整条链路"的设计），
但记 error 级 + 完整堆栈 + `degraded=true` 标记，可以在日志里查出来——
"静默"才是当初真正的问题，"降级"本身不是。

**为什么暂停在分类、不做重试**：README 的路线把「故障分类重试」放在 Phase 2。
这里只负责把类型分清、处置分明，重试策略等有了稳定的错误分布数据再加——
现在连"哪类错误出现过几次"都还统计不了。
"""
from enum import Enum
from typing import Any

from app.core.logger import logger

NODE_NAME = "error_policy"


class ErrorKind(str, Enum):
    """异常分类。继承 str 便于直接写进结构化日志的字段"""

    FATAL = "fatal"            # 代码自身不一致 → 上抛
    RETRYABLE = "retryable"    # 暂时性外部故障 → 降级，Phase 2 在此接重试
    BLOCKED = "blocked"        # 重试无用且需人处理 → 降级 + error
    UNEXPECTED = "unexpected"  # 兜底 → 降级 + error


# 由解释器抛出的、**只可能反映我们自己代码有问题**的异常类型。
# 这类异常与外部数据无关，重试和降级都没有意义——降级只会让它潜伏下来。
FATAL_TYPES = (
    NameError,
    UnboundLocalError,
    ImportError,
    NotImplementedError,
    AssertionError,
    SyntaxError,
    IndentationError,
    RecursionError,
)

# 名字里带这些词的异常按「暂时性」处理（外部服务抖动、网络问题）。
# 用名字而不是 isinstance：这些类型散落在 openai / httpx / pymongo / neo4j / milvus 里，
# 逐个 import 会把这些库变成硬依赖，而它们本就都是可选的外部服务。
RETRYABLE_HINTS = (
    "timeout", "timed out", "connection", "connect", "unavailable",
    "reconnect", "temporarily", "rate limit", "too many requests", "try again",
)

# 可重试的 HTTP 状态码：408 请求超时、425 过早、429 限流、5xx 服务端问题
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _http_status(exc: BaseException) -> int:
    """
    取 HTTP 状态码，取不到返回 0

    只认 HTTP 语义的字段：`openai` 系用 `status_code`，`httpx.HTTPStatusError` 在
    `response.status_code` 上。**刻意不读 `.code`** —— Milvus / Neo4j 的 `.code` 是
    各自内部的错误码（如 Milvus 的 1100），拿它当 HTTP 状态码会得出错误结论。
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else 0


def classify(exc: BaseException) -> ErrorKind:
    """
    给异常分类。顺序有讲究：先看是不是「我们自己的代码错」，再看外部状态

    :param exc: 捕获到的异常
    :return: 见 `ErrorKind`
    """
    # 1. 代码自身不一致：与外部无关，必须暴露
    if isinstance(exc, FATAL_TYPES):
        return ErrorKind.FATAL

    # 2. HTTP 状态码（最可靠的信号，有就优先用）
    status = _http_status(exc)
    if status:
        if status in RETRYABLE_STATUS:
            return ErrorKind.RETRYABLE
        if 400 <= status < 500:
            # 鉴权、参数、配额之类：重试不会有不同结果，要人去改配置/代码
            return ErrorKind.BLOCKED

    # 3. 按异常名判断暂时性故障（网络类异常的名字高度一致）
    signature = f"{type(exc).__name__} {exc}".lower()
    if any(hint in signature for hint in RETRYABLE_HINTS):
        return ErrorKind.RETRYABLE

    # 4. 兜底：可能是外部依赖的其它异常，也可能是数据结构不符预期。
    #    仍然降级（保住"一路坏不影响整条链路"），但要留下完整堆栈和可统计的标记
    return ErrorKind.UNEXPECTED


def is_fatal(exc: BaseException) -> bool:
    """是不是「代码自身不一致」那类异常"""
    return classify(exc) is ErrorKind.FATAL


def degrade(node: str, what: str, fallback: Any, exc: BaseException, **extra) -> Any:
    """
    降级处置：分类 → 记日志 → 返回兜底值；是编程错误则上抛

    用法（替换原来的 `except: 记日志 + return 空`）：

    ```python
    except Exception as e:
        return degrade(NODE_NAME, "图谱检索", {"kg_chunks": []}, e)
    ```

    **不必非在 `except` 块内调用**：FATAL 分支用 `raise exc` 而不是裸 `raise`，
    后者没有活动异常时会抛 `RuntimeError`（自测里就踩过），
    前者在两种上下文下都能把原异常连同原始堆栈一起抛出去。

    :param node: 节点名。上下文里有节点时会被日志补丁去掉（避免重复），
        命令行单跑节点时它会出现在消息里，正好补上缺失的归因
    :param what: 干了什么事失败，用于人读，如「图谱检索」「生成假设文档」
    :param fallback: 降级时返回的值
    :param exc: 捕获到的异常
    :param extra: 额外的结构化字段，会进日志的 `extra`
    :return: `fallback`（FATAL 时不会返回）
    """
    kind = classify(exc)
    detail = f"{type(exc).__name__}: {exc}"

    if kind is ErrorKind.FATAL:
        # 这条日志本身就是本模块存在的理由：以前它会是一条 warning，然后功能静默失效
        logger.error(
            f"[{node}] {what}遇到编程错误，已上抛（不降级掩盖）：{detail}",
            exc_info=True, degraded=False, kind=kind.value,
        )
        raise exc

    if kind is ErrorKind.RETRYABLE:
        # 暂时性故障：warn 级、不带堆栈（网络抖动的堆栈是噪音，Phase 2 会在这里接重试）
        logger.warning(
            f"[{node}] {what}失败，降级继续（{kind.value}，可重试）：{detail}",
            degraded=True, kind=kind.value,
        )
    else:
        # BLOCKED / UNEXPECTED：要人看，带完整堆栈
        logger.error(
            f"[{node}] {what}失败，降级继续（{kind.value}）：{detail}",
            exc_info=True, degraded=True, kind=kind.value,
        )

    return fallback


def degrade_dependency(node: str, what: str, fallback: Any, reason: str,
                       kind: ErrorKind = ErrorKind.RETRYABLE) -> Any:
    """
    前置检查发现依赖不可用时的降级（**没有异常对象可分类**）

    与 `degrade` 分开，是因为这不是"抛了异常"——是主动探测到依赖不在（Neo4j 没起、
    Milvus 连不上、集合名没配），于是**整段功能跳过**。但它同样属于"功能在静默失效"：
    图谱那一路整段没跑、产品名对齐整段没跑，用户只是觉得"答得不好"，没人会知道原因。

    所以照样打 `degraded` 标记，否则 `log_query --degraded` 会漏掉最常见的一类降级
    （实测就是这样发现的：Neo4j 停掉后那条 warning 查不出来）。

    :param kind: 默认 `RETRYABLE`（多半是服务没起/在重启，起来就好）；
        配置缺失这类"重试也没用、要人改"的传 `ErrorKind.BLOCKED`
    :return: `fallback`
    """
    logger.warning(
        f"[{node}] {what}跳过，降级继续（{kind.value}，依赖不可用）：{reason}",
        degraded=True, kind=kind.value, dependency=reason,
    )
    return fallback


if __name__ == '__main__':
    """自测：分类是否符合预期、degrade 的四种走向"""

    # 用假的异常类型模拟各来源，避免自测依赖外部库
    class FakeTimeout(Exception):
        pass

    class FakeAPIStatus(Exception):
        def __init__(self, status_code):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    class ServerSelectionTimeoutError(Exception):
        pass

    cases = [
        # 编程错误 → FATAL
        (NameError("name 'x' is not defined"), ErrorKind.FATAL),
        (UnboundLocalError("local variable 'y' referenced before assignment"), ErrorKind.FATAL),
        (ImportError("No module named 'foo'"), ErrorKind.FATAL),
        (AssertionError("不该发生"), ErrorKind.FATAL),
        # 暂时性 → RETRYABLE
        (TimeoutError("timed out"), ErrorKind.RETRYABLE),
        (FakeTimeout("read timeout"), ErrorKind.RETRYABLE),
        (ConnectionError("connection reset"), ErrorKind.RETRYABLE),
        (ServerSelectionTimeoutError("No servers found yet"), ErrorKind.RETRYABLE),
        (FakeAPIStatus(429), ErrorKind.RETRYABLE),
        (FakeAPIStatus(503), ErrorKind.RETRYABLE),
        # 重试无用 → BLOCKED
        (FakeAPIStatus(401), ErrorKind.BLOCKED),
        (FakeAPIStatus(400), ErrorKind.BLOCKED),
        # 兜底 → UNEXPECTED
        (ValueError("RERANK_API_KEY 未配置"), ErrorKind.UNEXPECTED),
        (KeyError("output"), ErrorKind.UNEXPECTED),
        (TypeError("unsupported operand"), ErrorKind.UNEXPECTED),
    ]

    problems = []
    for exc, want in cases:
        got = classify(exc)
        if got is not want:
            problems.append(f"{type(exc).__name__} 应为 {want.value}，实际 {got.value}")

    # degrade 的四种走向
    # FATAL 必须上抛，不能被降级吞掉
    try:
        degrade("node_x", "图谱检索", {"kg_chunks": []}, NameError("拼错的变量"))
        problems.append("NameError 被降级吞掉了，应上抛")
    except NameError:
        pass

    if degrade("node_x", "联网搜索", {"docs": []}, TimeoutError("timed out")) != {"docs": []}:
        problems.append("RETRYABLE 应返回兜底值")
    if degrade("node_x", "重排", [], ValueError("配置缺失")) != []:
        problems.append("UNEXPECTED 应返回兜底值")

    for p in problems:
        logger.error(f"[测试] [FAIL] {p}")
    if not problems:
        logger.success(f"[测试] [PASS] 异常分级验证通过（{len(cases)} 个分类 + 3 个降级走向）")

    # 结构化字段：degraded / kind 应能进 JSONL 的 extra
    degrade("node_x", "演示", [], TimeoutError("timeout"))
