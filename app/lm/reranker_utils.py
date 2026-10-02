"""
重排序工具（DashScope gte-rerank-v2）

与教程的差异：教程用本地 BGE（FlagEmbedding 的 FlagReranker），本项目改走 DashScope
重排 API。取舍同嵌入模型：项目已是全 DashScope API 架构（LLM / 嵌入 / 重排共用一个 key），
本地方案要额外装 torch（约 2GB）+ 下载 1.3GB 模型，且本机无 CUDA、只能 CPU 推理，
而重排是每次查询都要跑的环节。

**分数尺度提醒**：本地 BGE 返回无界 logits，本 API 返回 0~1 归一化分数。
调用方 node_rerank 里按 BGE 尺度设定的断崖阈值需据此重新审视。
"""
import httpx

from app.conf.reranker_config import reranker_config
from app.core.logger import logger
from app.core.usage_tracker import Timer, record

TIMEOUT = 60.0


class RerankError(Exception):
    """重排调用失败：配置缺失、接口报错或返回结构异常"""


def rerank(query: str, documents: list, top_n: int = None) -> list:
    """
    调 DashScope 重排模型，为「查询-文档」对打分

    :param query: 查询文本（通常是改写后的问题）
    :param documents: 待排序的文档正文列表
    :param top_n: 只返回前 N 条，None 表示全部
    :return: [{"index": 原文档下标, "score": 相关性分数}, ...]，按分数降序
    :raises RerankError: 配置缺失或接口报错
    """
    if not reranker_config.api_key:
        raise RerankError("RERANK_API_KEY / OPENAI_API_KEY 未配置")
    if not documents:
        return []

    payload = {
        "model": reranker_config.model,
        "input": {"query": query, "documents": documents},
        "parameters": {
            "return_documents": False,
            # 显式要全量：接口默认只回前若干条，而截断应交给 node_rerank 的动态 TopK 决定
            "top_n": top_n or len(documents),
        },
    }

    timer = Timer()
    try:
        with timer:
            resp = httpx.post(
                reranker_config.base_url,
                headers={
                    "Authorization": "Bearer " + reranker_config.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=TIMEOUT,
            )
    except Exception as e:
        # 记账后再抛：失败的调用同样入账，用于观察错误率
        record("rerank", model=reranker_config.model, latency_ms=timer.ms, ok=False,
               error=f"请求失败：{e}", docs=len(documents))
        raise RerankError(f"重排请求失败：{e}") from e

    if resp.status_code != 200:
        record("rerank", model=reranker_config.model, latency_ms=timer.ms, ok=False,
               error=f"HTTP {resp.status_code}", docs=len(documents))
        raise RerankError(f"重排接口返回 HTTP {resp.status_code}：{resp.text[:200]}")

    body = resp.json()
    results = (body.get("output") or {}).get("results")
    if results is None:
        record("rerank", model=reranker_config.model, latency_ms=timer.ms, ok=False,
               error="返回结构异常", docs=len(documents))
        raise RerankError(f"重排返回结构异常：{str(body)[:200]}")

    # 接口按 index 指回入参下标，这里一并带出，供调用方还原文档
    scored = [
        {"index": int(r["index"]), "score": float(r["relevance_score"])}
        for r in results
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)

    usage = body.get("usage") or {}
    total_tokens = usage.get("total_tokens") or 0
    record("rerank", model=reranker_config.model, prompt_tokens=total_tokens,
           latency_ms=timer.ms, docs=len(documents))
    logger.info(
        f"[重排] {len(scored)} 条打分完成，"
        f"分数区间 {scored[0]['score']:.4f}~{scored[-1]['score']:.4f}，"
        f"tokens={usage.get('total_tokens')}"
    )
    return scored


if __name__ == '__main__':
    """
    本地测试：验证重排接口连通性与打分合理性

    前置：.env 已配置 OPENAI_API_KEY（或 RERANK_API_KEY）
    """
    query = "烫金机怎么设置局部烫金区域"
    docs = [
        "打开电源开关，接入220V电源",
        "遇到故障请联系售后",
        "佩戴防护手套后再操作",
        "本机支持局部烫金，范围可在操作面板设置为 50-170mm",
    ]

    logger.info("=" * 70)
    logger.info("[测试] 开始验证 DashScope 重排接口")
    try:
        got = rerank(query, docs)
        for rank, item in enumerate(got, 1):
            logger.info(
                f"[测试] {rank}. idx={item['index']} score={item['score']:.4f} "
                f"{docs[item['index']][:34]!r}"
            )

        problems = []
        if len(got) != len(docs):
            problems.append(f"返回条数不符：{len(got)} != {len(docs)}")
        elif got[0]["index"] != 3:
            problems.append(
                f"排序不合理：Top1 应为 idx=3（局部烫金范围），实际 idx={got[0]['index']}"
            )
        if any(i["index"] < 0 or i["index"] >= len(docs) for i in got):
            problems.append("返回了越界的 index")

        for p in problems:
            logger.error(f"[测试] [FAIL] {p}")
        if not problems:
            logger.success("[测试] [PASS] 重排接口验证通过")
    except Exception as e:
        logger.error(f"[测试] [FAIL] 调用失败：{e}", exc_info=True)
    logger.info("=" * 70)
