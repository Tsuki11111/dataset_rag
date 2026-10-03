"""
结构化日志查询（读 `logs/app_年月日.jsonl`）

日志有三个读者：控制台、`.log` 文本、`.jsonl` 结构化。前两个用眼睛看，
第三个要**查**——这个模块就是那个查询入口，省得为了一条日志装 jq。

用法：

```bash
# 看某一次请求的全部日志（一次问答 = 一个 trace，与账本的 trace_id 相同）
.venv/Scripts/python.exe -m app.core.log_query --trace 3dd63c53004e43c2

# 只看图检索节点出的错，最近 3 天
.venv/Scripts/python.exe -m app.core.log_query --node node_query_kg --level ERROR --days 3

# 按关键词搜（消息里包含就算命中）
.venv/Scripts/python.exe -m app.core.log_query --grep 重排 --days 7 --limit 50

# 只看降级事件：某一路召回/依赖失败后链路继续跑的那些（判断"哪个功能在悄悄失效"）
.venv/Scripts/python.exe -m app.core.log_query --degraded --days 7

# 保持原始 JSON（喂给别的工具）
.venv/Scripts/python.exe -m app.core.log_query --trace 3dd63c53004e43c2 --json
```

装过 jq 的话，等价的写法是：
`jq -c 'select(.trace_id=="...")' logs/app_20261003.jsonl`
"""
import argparse
import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# 日志目录：与 logger.py 保持一致（都取项目根下的 logs）
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_GLOB = "app_*.jsonl"


def _log_files(days: float) -> List[Path]:
    """按日期筛日志文件。文件名就是日期，不必读内容"""
    since = datetime.now() - timedelta(days=days)
    files = []
    for path in glob.glob(str(LOG_DIR / LOG_GLOB)):
        stamp = Path(path).stem.replace("app_", "")
        try:
            if datetime.strptime(stamp, "%Y%m%d") >= since.replace(hour=0, minute=0, second=0, microsecond=0):
                files.append(Path(path))
        except ValueError:
            # 文件名不合约定（比如手工改名过），跳过而不是崩
            continue
    return sorted(files)


def _match(item: Dict[str, Any], args) -> bool:
    if args.trace and item.get("trace_id") != args.trace:
        return False
    if args.node and item.get("node") != args.node:
        return False
    if args.level and item.get("level") != args.level.upper():
        return False
    if args.grep and args.grep not in (item.get("message") or ""):
        return False
    if args.degraded:
        # 降级事件由 error_policy.degrade 打上 degraded=true；
        # 有了这个开关，"哪条召回路径在悄悄降级"才查得出来
        extra = item.get("extra") or {}
        if not extra.get("degraded"):
            return False
    return True


def _format(item: Dict[str, Any]) -> str:
    """人读格式：时间 | 级别 | 节点 | 位置 - 消息"""
    ts = (item.get("ts") or "")[:23].replace("T", " ")
    node = item.get("node") or "—"
    where = f"{item.get('module', '?')}:{item.get('line', '?')}"
    line = f"{ts} | {item.get('level', ''):<7} | {node:<26} | {where:<28} - {item.get('message', '')}"
    if item.get("exception"):
        line += f"\n    └─ {item['exception']['type']}: {item['exception']['message']}"
    return line


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.core.log_query",
        description="查询结构化日志（logs/*.jsonl）",
    )
    parser.add_argument("--trace", help="按 trace_id 精确查（一次问答/一次导入）")
    parser.add_argument("--node", help="按图节点名精确查，如 node_rerank")
    parser.add_argument("--level", help="按级别查：INFO / WARNING / ERROR")
    parser.add_argument("--grep", help="消息包含该子串")
    parser.add_argument("--degraded", action="store_true",
                        help="只看降级事件（某一路召回/依赖失败后继续跑的记录）")
    parser.add_argument("--days", type=float, default=1.0, help="回看天数，默认 1（今天）")
    parser.add_argument("--limit", type=int, default=200, help="最多输出多少条，默认 200")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON 行而不是人读格式")
    args = parser.parse_args(argv)

    files = _log_files(args.days)
    if not files:
        print(f"没有找到日志文件：{LOG_DIR / LOG_GLOB}（回看 {args.days:g} 天）")
        return 1

    hits, total = [], 0
    for path in files:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                # 单行坏了不该让整个查询失败（比如机器写日志时断电）
                continue
            total += 1
            if _match(item, args):
                hits.append(item)

    if not hits:
        print(f"扫描 {len(files)} 个文件、{total} 条日志，没有匹配的记录。")
        return 0

    # 时间升序：排查问题时按发生顺序读
    hits.sort(key=lambda x: x.get("ts") or "")
    shown = hits[-args.limit:]   # limit 取**最后** N 条（最近的），更符合排查习惯
    for item in shown:
        print(json.dumps(item, ensure_ascii=False) if args.json else _format(item))

    print(f"\n匹配 {len(hits)} 条" + (f"，显示其中最近 {len(shown)} 条" if len(hits) > len(shown) else "")
          + f"（扫描 {len(files)} 个文件、{total} 条）")
    return 0


if __name__ == '__main__':
    sys.exit(main())
