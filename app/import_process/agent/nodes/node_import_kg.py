"""
知识图谱导入节点 (node_import_kg)

把文档切片交给 LLM 抽取「实体 + 实体间关系」，连同切片挂载一起写入 Neo4j，
供将来的检索侧 node_query_kg 查询。

**必须排在 node_import_milvus 之后**：图谱挂载依赖 chunk_id，
而 chunk_id 是 Milvus 自增主键，由该节点写入后回填。

降级原则：图谱是补充召回，Neo4j 不可用或某批抽取失败都只记日志，**不中断文档导入**。
"""
import json
import sys

from langchain.messages import HumanMessage, SystemMessage

from app.clients.neo4j_utils import (
    ENTITY_TYPES,
    RELATION_TYPES,
    ensure_neo4j_schema,
    is_neo4j_available,
    write_doc_graph,
)
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_import_kg"

# 每批送给 LLM 的切片数。批越大调用越少，但越容易超长或漏条
BATCH_SIZE = 5
# 单个切片正文的截断长度，避免个别超长切片把整批撑爆
MAX_CHUNK_CHARS = 800


def step_1_get_inputs(state: ImportGraphState):
    """
    步骤 1: 取输入

    图谱挂载必须有 chunk_id，所以没有 chunk_id 或正文为空的切片直接跳过。
    :return: (file_title, item_name, 可用切片列表)
    """
    function_name = sys._getframe().f_code.co_name
    file_title = (state.get("file_title") or "").strip()
    item_name = (state.get("item_name") or "").strip()
    raw_chunks = state.get("chunks") or []

    usable = [
        c for c in raw_chunks
        if isinstance(c, dict) and c.get("chunk_id") and (c.get("content") or "").strip()
    ]
    skipped = len(raw_chunks) - len(usable)
    if skipped:
        logger.warning(
            f"[{NODE_NAME}] [{function_name}] 有 {skipped} 个切片缺少 chunk_id 或正文，已跳过"
        )
    logger.info(
        f"[{NODE_NAME}] [{function_name}] 入参：file_title={file_title!r}，"
        f"item_name={item_name!r}，可用切片 {len(usable)} 条"
    )
    return file_title, item_name, usable


def step_2_build_batches(chunks: list) -> list:
    """步骤 2: 按 BATCH_SIZE 切片分批"""
    return [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]


def step_3_extract_batch(batch: list, item_name: str) -> list:
    """
    步骤 3: 抽取单批切片

    :return: LLM 返回的 chunks 列表（每项含 chunk_index / entities / relations）
    :raises: JSON 解析失败或调用异常
    """
    chunk_text = "\n\n".join(
        f"--- 切片 {i} ---\n"
        f"标题：{c.get('title') or '(无)'}\n"
        f"内容：{(c.get('content') or '')[:MAX_CHUNK_CHARS]}"
        for i, c in enumerate(batch)
    )
    user_prompt = load_prompt(
        "kg_extraction", product=item_name or "未知产品", chunks=chunk_text
    )
    messages = [
        SystemMessage(load_prompt("kg_extraction_system")),
        HumanMessage(user_prompt),
    ]
    resp = get_llm_client(json_mode=True).invoke(messages)
    raw = (getattr(resp, "content", "") or "").strip()
    data = json.loads(raw)
    return data.get("chunks") or []


def step_4_extract_all(batches: list, item_name: str) -> list:
    """
    步骤 4: 逐批抽取，返回与 chunks 一一对应的结果列表

    单批失败只把该批位置留空，不影响其余批次。
    LLM 漏条或 chunk_index 越界时按位置补空，保证下游对齐不错位。
    """
    function_name = sys._getframe().f_code.co_name
    results = []

    for bi, batch in enumerate(batches, start=1):
        try:
            per_chunk = step_3_extract_batch(batch, item_name)
        except Exception as e:
            logger.error(
                f"[{NODE_NAME}] [{function_name}] 第 {bi}/{len(batches)} 批抽取失败，"
                f"该批 {len(batch)} 个切片跳过：{e}"
            )
            per_chunk = []

        aligned = [{} for _ in batch]
        for item in per_chunk:
            try:
                k = int(item.get("chunk_index"))
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(batch):
                aligned[k] = item
        results.extend(aligned)

    logger.info(f"[{NODE_NAME}] [{function_name}] 抽取完成，覆盖 {len(results)} 个切片")
    return results


def step_5_collect(extractions: list, chunks: list):
    """
    步骤 5: 汇总为实体 / 挂载 / 关系三张表，并修复悬空关系

    **修复点**：LLM 偶尔会在 relations 里引用没登记进 entities 的名称，
    而 Cypher 的 MATCH 找不到端点会**静默丢边**。与其丢掉，不如把端点补登记成实体
    （type 记「其他」）。补得过多说明提示词在漂移，所以把数量打出来。

    :return: (entities: {name: type}, links: [(name, chunk_id)], rels: [(src, dst, type)])
    """
    function_name = sys._getframe().f_code.co_name
    entities = {}
    links = []
    rels = []
    repaired = 0

    for chunk, extraction in zip(chunks, extractions):
        chunk_id = str(chunk["chunk_id"])
        local_names = set()

        for e in extraction.get("entities") or []:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            etype = e.get("type") if e.get("type") in ENTITY_TYPES else "其他"
            local_names.add(name)
            entities.setdefault(name, etype)
            links.append((name, chunk_id))

        for r in extraction.get("relations") or []:
            src = (r.get("src") or "").strip()
            dst = (r.get("dst") or "").strip()
            if not src or not dst or src == dst:
                continue
            rtype = r.get("type") if r.get("type") in RELATION_TYPES else "其他"
            # 端点必须在本切片的实体集合里，否则 Cypher 会丢边
            for endpoint in (src, dst):
                if endpoint not in local_names:
                    repaired += 1
                    local_names.add(endpoint)
                    entities.setdefault(endpoint, "其他")
                    links.append((endpoint, chunk_id))
            rels.append((src, dst, rtype))

    if repaired:
        logger.warning(
            f"[{NODE_NAME}] [{function_name}] 有 {repaired} 个关系端点未登记为实体，"
            f"已补记为「其他」类型（数量偏多说明提示词在漂移）"
        )

    # 去重（保持顺序）
    links = list(dict.fromkeys(links))
    rels = list(dict.fromkeys(rels))
    logger.info(
        f"[{NODE_NAME}] [{function_name}] 汇总：实体 {len(entities)} 个，"
        f"挂载 {len(links)} 条，关系 {len(rels)} 条"
    )
    return entities, links, rels


def step_6_persist(file_title, item_name, chunks, entities, links, rels) -> dict:
    """步骤 6: 写入 Neo4j 并对账"""
    function_name = sys._getframe().f_code.co_name
    result = write_doc_graph(
        file_title=file_title,
        item_name=item_name,
        chunks=[
            {"chunk_id": str(c["chunk_id"]),
             "title": c.get("title") or "",
             "parent_title": c.get("parent_title") or ""}
            for c in chunks
        ],
        entities=[{"name": n, "type": t} for n, t in entities.items()],
        links=[{"name": n, "chunk_id": cid} for n, cid in links],
        rels=[{"src": s, "dst": d, "type": t} for s, d, t in rels],
    )
    if result.get("ok"):
        stats = result["stats"]
        # 写入量少于待写量说明有边被丢，值得告警
        if stats.get("relations", 0) < len(rels):
            logger.warning(
                f"[{NODE_NAME}] [{function_name}] 关系写入数({stats.get('relations')}) "
                f"少于待写数({len(rels)})，可能有端点未落库"
            )
    return result


def node_import_kg(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入知识图谱 (node_import_kg)

    :param state: 需包含 task_id / file_title / item_name / chunks（含 chunk_id）
    :return: 空字典（图谱数据直接写库，不回写 state）
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    add_running_task(state.get("task_id", ""), function_name)

    try:
        # 前置检查：Neo4j 不可用就整段跳过，不能拖垮文档导入
        if not is_neo4j_available():
            logger.warning(f"[{NODE_NAME}] [{function_name}] Neo4j 不可用，跳过图谱构建")
            return {}
        ensure_neo4j_schema()

        file_title, item_name, chunks = step_1_get_inputs(state)
        if not file_title or not chunks:
            logger.warning(
                f"[{NODE_NAME}] [{function_name}] 缺少 file_title 或可用切片，跳过图谱构建"
            )
            return {}

        batches = step_2_build_batches(chunks)
        logger.info(f"[{NODE_NAME}] [{function_name}] 共 {len(chunks)} 个切片，分 {len(batches)} 批抽取")

        extractions = step_4_extract_all(batches, item_name)
        entities, links, rels = step_5_collect(extractions, chunks)
        result = step_6_persist(file_title, item_name, chunks, entities, links, rels)

        if result.get("ok"):
            logger.success(f"[{NODE_NAME}] [{function_name}] 图谱构建完成：{result['stats']}")
        else:
            logger.error(f"[{NODE_NAME}] [{function_name}] 图谱写入失败：{result.get('error')}")
        return {}

    except Exception as e:
        # 图谱失败不影响文档本身已入库的切片，所以不 raise
        logger.error(f"[{NODE_NAME}] [{function_name}] 图谱构建异常：{e}", exc_info=True)
        return {}
    finally:
        add_done_task(state.get("task_id", ""), function_name)
        logger.info(f"[{NODE_NAME}] [{function_name}] 处理结束")


if __name__ == '__main__':
    """
    本地测试：喂假切片跑完整抽取与落库（不需要 Milvus，只需要 Neo4j）

    前置：docker compose -f docker/neo4j-compose.yml up -d
    """
    from app.clients.neo4j_utils import count_graph_by_file_titles, delete_doc_graph
    from app.import_process.agent.state import create_default_state
    from app.utils.task_utils import clear_task

    TEST_FILE = "kg_节点测试文档"
    task_id = "kg_node_test"

    chunks = [
        {"chunk_id": "kg_node_1", "title": "## 3.4.2 装入半幅烫金膜盒", "parent_title": "# 操作",
         "content": "打开烫金膜盒支架盖。将半幅烫金膜盒沿着导轨推入，向下轻推，直到它锁定到位，听到咔哒声即表示安装完成。"},
        {"chunk_id": "kg_node_2", "title": "## 更换电池", "parent_title": "# 维护",
         "content": "关闭设备电源并拔掉电源适配器。用十字螺丝刀卸下后盖上的4颗螺丝，取出旧电池，更换为同型号电池后重新装回后盖。"},
        {"chunk_id": "kg_node_3", "title": "## 目录", "parent_title": "# 前言",
         "content": "前言 / 安全须知 / 使用设备 / 维护保养"},
    ]

    st = create_default_state(task_id=task_id, file_title=TEST_FILE,
                              item_name="Brother HAK 180 烫金机", chunks=chunks)

    logger.info("=" * 70)
    logger.info("[测试] 开始验证 node_import_kg")
    problems = []

    try:
        node_import_kg(st)
        counts = count_graph_by_file_titles([TEST_FILE]).get(TEST_FILE, {})
        logger.info(f"[测试] 图谱统计：{counts}")

        if counts.get("chunks") != len(chunks):
            problems.append(f"Chunk 数不符：{counts.get('chunks')} != {len(chunks)}")
        if not counts.get("entities"):
            problems.append("没有抽到任何实体")
        if counts.get("entities", 0) < 5:
            problems.append(f"实体数偏少（{counts.get('entities')}），抽取可能失败")

        # 幂等：再跑一次，图谱不应累积。
        # 注意实体数会有小幅波动——抽取走 LLM，两次调用结果本就不完全一致；
        # 写库本身的幂等已在 neo4j_utils 自测里用固定数据精确验证过，
        # 这里只校验「没有累积」这个端到端性质。
        node_import_kg(create_default_state(task_id=task_id, file_title=TEST_FILE,
                                            item_name="Brother HAK 180 烫金机", chunks=chunks))
        counts2 = count_graph_by_file_titles([TEST_FILE]).get(TEST_FILE, {})
        logger.info(f"[测试] 二次运行后统计：{counts2}")
        if counts2.get("chunks") != len(chunks):
            problems.append(f"二次运行 Chunk 数不符：{counts2.get('chunks')} != {len(chunks)}")
        if counts2.get("entities", 0) >= counts.get("entities", 0) * 2:
            problems.append(f"图谱出现累积（幂等失败）：{counts} -> {counts2}")
    except Exception as e:
        problems.append(f"执行异常：{e}")
    finally:
        delete_doc_graph(TEST_FILE)
        clear_task(task_id)

    for p in problems:
        logger.error(f"[测试] [FAIL] {p}")
    if not problems:
        logger.success("[测试] [PASS] node_import_kg 验证通过（含幂等）")
    logger.info("=" * 70)
