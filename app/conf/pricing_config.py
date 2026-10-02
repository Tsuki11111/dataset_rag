"""
模型计价表（估算调用成本用）

**单位：元 / 百万 tokens**，取自阿里云百炼官方价目表（华北2·北京，按量付费）。
来源与查询日期写在每条价格后面，改了价格记得一起改这一行。

**价格只是快照，不是事实**：账本里同时记了 tokens，所以单价变了随时能按新价重算
（`app/clients/mongo_usage_utils.py` 的报表就是现算的，不读历史成本字段）——
tokens 才是原始事实，成本是派生值。

三处刻意的简化，都写在明面上而不是藏着：

- **qwen-plus 按上下文长度阶梯计价**（≤128k: 0.8/2；128k~256k: 2.4/20；256k~1m: 4.8/48）。
  本项目单次请求的上下文是若干切片，远小于 128k，因此统一按最低档计。
  真要处理超长上下文，得把 `PRICE_PER_MILLION` 换成按输入区间分档的结构。
- **联网搜索（百炼 MCP EnhancedSearch）按次计费**，单价未见于公开文档，
  故账本里只记调用次数、不计成本（`estimate_cost` 返回 None）。
- **MinerU 解析按页数配额计费**，与 token 无关，不在这张表里。
"""
from typing import Optional

# 型号 → {input: 每百万输入 token 单价, output: 每百万输出 token 单价}
# 仅输入计费的模型（嵌入 / 重排）output 记 0。
PRICE_PER_MILLION = {
    # 百炼文档：qwen-plus（2026-10 查询）
    "qwen-plus": {"input": 0.8, "output": 2.0},
    # 百炼文档：qwen3-vl-flash（2026-10 查询）
    "qwen3-vl-flash": {"input": 0.15, "output": 1.5},
    # 百炼文档：text-embedding-v2，0.0007 元/千 token（2026-10 查询）
    "text-embedding-v2": {"input": 0.7, "output": 0.0},
    # 百炼文档：gte-rerank-v2，0.8 元/百万 tokens（2026-10 查询）
    "gte-rerank-v2": {"input": 0.8, "output": 0.0},
}


def _lookup(model: str) -> Optional[dict]:
    """
    查单价，兼容带日期后缀的型号名

    DashScope 既接受 `qwen-plus` 也接受 `qwen-plus-2025-04-28` 这类带日期的快照名，
    两者同价，所以先精确匹配、再去掉日期后缀匹配。
    """
    if not model:
        return None
    if model in PRICE_PER_MILLION:
        return PRICE_PER_MILLION[model]

    # 去掉形如 -2025-04-28 的日期后缀再试一次
    parts = model.rsplit("-", 3)
    if len(parts) == 4 and all(p.isdigit() for p in parts[1:]):
        return PRICE_PER_MILLION.get(parts[0])
    return None


def estimate_cost(model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> Optional[float]:
    """
    按 tokens 估算一次调用的成本（元）

    :param model: 模型名
    :param prompt_tokens: 输入 tokens
    :param completion_tokens: 输出 tokens
    :return: 估算成本（元）；型号不在表里时返回 None —— 宁可留空，也不要编一个数
    """
    price = _lookup(model)
    if not price:
        return None

    cost = (
        (prompt_tokens or 0) * price["input"]
        + (completion_tokens or 0) * price["output"]
    ) / 1_000_000
    return round(cost, 8)


if __name__ == '__main__':
    """自测：验证计价与型号名兼容"""
    from app.core.logger import logger

    cases = [
        ("qwen-plus", 3000, 500, True),
        ("qwen-plus-2025-04-28", 3000, 500, True),   # 带日期后缀应同价
        ("text-embedding-v2", 10000, 0, True),
        ("gte-rerank-v2", 2000, 0, True),
        ("some-unknown-model", 100, 100, False),     # 未知型号应返回 None
    ]
    problems = []
    for model, pin, pout, should_have_price in cases:
        cost = estimate_cost(model, pin, pout)
        ok = (cost is not None) == should_have_price
        logger.info(f"[测试] {model:<24} in={pin:<6} out={pout:<5} cost={cost} {'OK' if ok else 'FAIL'}")
        if not ok:
            problems.append(model)

    if problems:
        logger.error(f"[测试] [FAIL] 行为不符预期：{problems}")
    else:
        logger.success("[测试] [PASS] 计价表验证通过")
