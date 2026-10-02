"""
调用账本（MongoDB 集合 `llm_usage`）

`app/core/usage_tracker.py` 每记一笔就写一条，**append-only，不做原地更新**。
一次问答要调十几次外部模型，逐次记账才能回答「钱花在哪」——是按类型（LLM 生成 vs
嵌入 vs 重排 vs 联网）、按模型、按租户，还是按某次特别贵的问答。

两个刻意的设计：

- **只存 tokens，不存死成本**。每条记录里确实带了当时的成本估算，但报表一律**按当前
  计价表重算**（`estimate_cost`）。单价会变（qwen-plus 就调过价），tokens 是原始事实、
  成本是派生值——重算才能让历史账目跟着新价走，而不是冻结在当初的估算上。
- **失败也入账**（`ok=false` + 错误摘要）。错误率与延迟分布和成本同样重要，
  而且「哪条路经常挂」这类问题只有失败记录能回答。

为什么落 MongoDB 而不是新引一个时序库：项目已经用它存会话历史与去重记录，
一份账本不值得再加一个要运维的服务。

用法：
    .venv/Scripts/python.exe -m app.clients.mongo_usage_utils        # 最近 1 天
    .venv/Scripts/python.exe -m app.clients.mongo_usage_utils 7      # 最近 7 天
"""
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo import DESCENDING, MongoClient

from app.conf.pricing_config import estimate_cost
from app.core.logger import logger

# 集合名
COLLECTION_NAME = "llm_usage"

_usage_tool = None


class UsageTool:
    """账本读写工具类：封装连接、集合与索引（沿用 mongo_user_utils 的写法）"""

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.collection = self.db[COLLECTION_NAME]

            # 按时间查账（报表默认路径）
            self.collection.create_index([("ts", DESCENDING)])
            # 按一次请求（trace）聚合，看单次问答成本
            self.collection.create_index([("trace_id", 1)])
            # 按租户归集成本
            self.collection.create_index([("tenant_id", 1), ("ts", DESCENDING)])
        except Exception as e:
            logger.error(f"[账本] 初始化失败：{e}", exc_info=True)
            raise


def get_usage_tool() -> UsageTool:
    """获取账本工具单例（懒加载）"""
    global _usage_tool
    if _usage_tool is None:
        _usage_tool = UsageTool()
    return _usage_tool


def save_usage(item: Dict[str, Any]) -> None:
    """
    写一条账目。调用方（usage_tracker）已负责吞异常，这里再兜一层，保证不会向上抛
    """
    try:
        doc = dict(item)
        # ts 另存一份可读时间，方便直接看库（毫秒时间戳本身不直观）
        doc["datetime"] = datetime.fromtimestamp(doc.get("ts") or time.time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        get_usage_tool().collection.insert_one(doc)
    except Exception as e:
        logger.warning(f"[账本] 写入失败（不影响业务）：{e}")


def _since(days: float) -> float:
    """最近 N 天的时间戳下界"""
    return (datetime.now() - timedelta(days=days)).timestamp()


def _cost_of(row: Dict[str, Any]) -> float:
    """按当前计价表重算这条账目的成本（不在表里则记 0，并单独计数）"""
    cost = estimate_cost(
        row.get("model") or "",
        row.get("prompt_tokens") or 0,
        row.get("completion_tokens") or 0,
    )
    return cost or 0.0


def build_report(days: float = 1.0, top: int = 5) -> Dict[str, Any]:
    """
    汇总最近 N 天的账目

    :param days: 回看天数
    :param top: 列出最贵的若干次请求
    :param 返回：{overall, by_kind, by_model, by_tenant, top_traces, unpriced}
    """
    rows = list(
        get_usage_tool().collection.find({"ts": {"$gte": _since(days)}}).sort("ts", DESCENDING)
    )

    overall = {
        "calls": len(rows),
        "failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "avg_latency_ms": 0,
    }
    by_kind: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "cost": 0.0, "tokens": 0, "latency_ms": 0, "failed": 0}
    )
    by_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "cost": 0.0, "tokens": 0, "failed": 0}
    )
    by_tenant: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    traces: Dict[str, Dict[str, Any]] = {}
    unpriced = 0
    latency_sum = 0

    for row in rows:
        cost = _cost_of(row)
        pt = row.get("prompt_tokens") or 0
        ct = row.get("completion_tokens") or 0
        ok = row.get("ok", True)
        latency = row.get("latency_ms") or 0

        if row.get("model") and estimate_cost(row["model"], 1, 0) is None:
            # 单价查不到（如联网搜索按次计费），成本按 0 计但单独计数，避免误以为它免费
            unpriced += 1

        overall["prompt_tokens"] += pt
        overall["completion_tokens"] += ct
        overall["cost"] += cost
        overall["failed"] += 0 if ok else 1
        latency_sum += latency

        kind = row.get("kind") or "unknown"
        by_kind[kind]["calls"] += 1
        by_kind[kind]["cost"] += cost
        by_kind[kind]["tokens"] += pt + ct
        by_kind[kind]["latency_ms"] += latency
        by_kind[kind]["failed"] += 0 if ok else 1

        model = row.get("model") or "—"
        by_model[model]["calls"] += 1
        by_model[model]["cost"] += cost
        by_model[model]["tokens"] += pt + ct
        by_model[model]["failed"] += 0 if ok else 1

        tenant = row.get("tenant_id") or "（未标注）"
        by_tenant[tenant]["calls"] += 1
        by_tenant[tenant]["cost"] += cost

        trace_id = row.get("trace_id") or "（无 trace）"
        t = traces.setdefault(
            trace_id,
            {
                "trace_id": trace_id,
                "tenant_id": row.get("tenant_id") or "",
                "session_id": row.get("session_id") or "",
                "calls": 0,
                "cost": 0.0,
                "tokens": 0,
                "ts": row.get("ts") or 0,
                "datetime": row.get("datetime") or "",
            },
        )
        t["calls"] += 1
        t["cost"] += cost
        t["tokens"] += pt + ct
        if (row.get("ts") or 0) < t["ts"]:
            t["ts"] = row.get("ts")
            t["datetime"] = row.get("datetime") or ""

    overall["avg_latency_ms"] = int(latency_sum / len(rows)) if rows else 0
    overall["cost"] = round(overall["cost"], 6)

    for k in by_kind:
        by_kind[k]["cost"] = round(by_kind[k]["cost"], 6)
        by_kind[k]["avg_latency_ms"] = (
            int(by_kind[k]["latency_ms"] / by_kind[k]["calls"]) if by_kind[k]["calls"] else 0
        )
    for k in by_model:
        by_model[k]["cost"] = round(by_model[k]["cost"], 6)
    for k in by_tenant:
        by_tenant[k]["cost"] = round(by_tenant[k]["cost"], 6)

    top_traces = sorted(traces.values(), key=lambda t: t["cost"], reverse=True)[:top]
    for t in top_traces:
        t["cost"] = round(t["cost"], 6)

    return {
        "days": days,
        "overall": overall,
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: kv[1]["cost"], reverse=True)),
        "by_model": dict(sorted(by_model.items(), key=lambda kv: kv[1]["cost"], reverse=True)),
        "by_tenant": dict(sorted(by_tenant.items(), key=lambda kv: kv[1]["cost"], reverse=True)),
        "top_traces": top_traces,
        "unpriced_calls": unpriced,
    }


def summarize_trace(trace_id: str) -> Dict[str, Any]:
    """看某一次请求（一次问答）的明细，排查「这次怎么这么贵」"""
    rows = list(get_usage_tool().collection.find({"trace_id": trace_id}).sort("ts", 1))
    for r in rows:
        r["_id"] = str(r.get("_id"))
        r["cost"] = _cost_of(r)
    return {
        "trace_id": trace_id,
        "calls": len(rows),
        "cost": round(sum(r["cost"] for r in rows), 6),
        "items": rows,
    }


def _print_report(days: float) -> None:
    """命令行报表：不引表格库，手工对齐即可"""
    report = build_report(days)
    o = report["overall"]

    print(f"\n=== 调用账本 · 最近 {days:g} 天 ===")
    if not o["calls"]:
        print("  还没有任何记录。跑一次问答或导入就会产生。\n")
        return

    print(f"  调用总数 : {o['calls']}（失败 {o['failed']}）")
    print(f"  tokens   : 输入 {o['prompt_tokens']} + 输出 {o['completion_tokens']}"
          f" = {o['prompt_tokens'] + o['completion_tokens']}")
    print(f"  估算成本 : {o['cost']:.6f} 元（按当前计价表重算）")
    print(f"  平均延迟 : {o['avg_latency_ms']} ms")
    if report["unpriced_calls"]:
        print(f"  未计价   : {report['unpriced_calls']} 次调用不在计价表里，成本按 0 计")

    print("\n  ── 按类型 ──")
    for kind, s in report["by_kind"].items():
        share = (s["cost"] / o["cost"] * 100) if o["cost"] else 0
        print(f"  {kind:<12} 调用 {s['calls']:<5} tokens {s['tokens']:<9} "
              f"成本 {s['cost']:.6f} 元 ({share:5.1f}%)  平均 {s['avg_latency_ms']}ms"
              + (f"  失败 {s['failed']}" if s["failed"] else ""))

    print("\n  ── 按模型 ──")
    for model, s in report["by_model"].items():
        print(f"  {model:<22} 调用 {s['calls']:<5} tokens {s['tokens']:<9} 成本 {s['cost']:.6f} 元"
              + (f"  失败 {s['failed']}" if s["failed"] else ""))

    if len(report["by_tenant"]) > 1 or "（未标注）" not in report["by_tenant"]:
        print("\n  ── 按租户 ──")
        for tenant, s in report["by_tenant"].items():
            print(f"  {tenant:<22} 调用 {s['calls']:<5} 成本 {s['cost']:.6f} 元")

    if report["top_traces"]:
        print("\n  ── 最贵的几次请求 ──")
        for t in report["top_traces"]:
            print(f"  {t['datetime'] or '—':<20} {t['cost']:.6f} 元  "
                  f"调用 {t['calls']:<3} tokens {t['tokens']:<8} "
                  f"session={t['session_id'] or '—'}")
    print(f"\n  看某次请求的明细：summarize_trace('<trace_id>')\n")


if __name__ == '__main__':
    days_arg = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    _print_report(days_arg)
