"""
Neo4j 客户端与知识图谱读写工具

图谱结构（导入侧 node_import_kg 写入，将来检索侧 node_query_kg 读取）：
    (:Entity {name, type, item_name, file_title})
    (:Chunk  {chunk_id, title, parent_title, item_name, file_title})
    (:Entity)-[:APPEARS_IN]->(:Chunk)
    (:Entity)-[r:REL {type, file_title}]->(:Entity)

两个必须遵守的约定：

1. **每个 Entity / Chunk 都带 file_title，不存在跨文档共享节点。**
   这是 delete_doc_graph 用 DETACH DELETE 不会误删其他文档的全部依据。
   将来若要做跨文档实体合并，这套清理会立刻失效，必须同步改成「按文档记录拥有关系再删」。

2. **关系类型是 :REL 上的 `type` 属性，不是关系标签。**
   Cypher 无法参数化关系标签，拼字符串既有注入风险又无法约束取值范围，
   所以统一用单标签 + 属性，取值在 Python 侧按白名单兜底。
"""
import os
from typing import Any, Dict, List

from neo4j import GraphDatabase

from app.core.logger import logger

# 实体类型白名单；LLM 越界的取值会被兜底成「其他」
ENTITY_TYPES = ("部件", "操作", "故障", "参数", "其他")
# 关系类型白名单；同上
RELATION_TYPES = ("组成", "导致", "解决", "使用", "连接", "参数属于", "其他")

_driver = None


def get_neo4j_driver():
    """获取 Neo4j 驱动实例（单例）"""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
    return _driver


# ========================
# 约束与索引
# ========================
_SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE (n.name, n.file_title) IS UNIQUE",
    "CREATE CONSTRAINT chunk_key IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE INDEX entity_file_title IF NOT EXISTS FOR (n:Entity) ON (n.file_title)",
    "CREATE INDEX chunk_file_title IF NOT EXISTS FOR (n:Chunk) ON (n.file_title)",
    # 检索侧按 item_name 跨文档查（图谱按 file_title 隔离写入，但查询应按产品聚合）
    "CREATE INDEX entity_item_name IF NOT EXISTS FOR (n:Entity) ON (n.item_name)",
    "CREATE INDEX chunk_item_name IF NOT EXISTS FOR (n:Chunk) ON (n.item_name)",
)


# ========================
# Cypher：清理
# ========================
# 先单独删关系：REL 自带 file_title，即使将来某个端点变成共享节点也能清干净
_DELETE_RELATIONS = "MATCH ()-[r:REL]->() WHERE r.file_title = $ft DELETE r"
# DETACH DELETE 会一并带走该节点的所有边，无残留
_DELETE_NODES = "MATCH (n) WHERE n.file_title = $ft DETACH DELETE n"


# ========================
# Cypher：写入
# ========================
# Chunk 只按 chunk_id 做 key（它是 Milvus 自增主键，全局唯一），file_title 走 SET
_WRITE_CHUNKS = """
UNWIND $chunks AS c
MERGE (k:Chunk {chunk_id: c.chunk_id})
SET k.title = c.title, k.parent_title = c.parent_title,
    k.file_title = $ft, k.item_name = $item
"""

_WRITE_ENTITIES = """
UNWIND $entities AS e
MERGE (n:Entity {name: e.name, file_title: $ft})
ON CREATE SET n.type = e.type
SET n.item_name = $item
"""

_WRITE_LINKS = """
UNWIND $links AS l
MATCH (e:Entity {name: l.name, file_title: $ft})
MATCH (c:Chunk {chunk_id: l.chunk_id})
MERGE (e)-[:APPEARS_IN]->(c)
"""

# MERGE 的 key 必须含 type，否则同一对实体之间只能存下一种关系
_WRITE_RELATIONS = """
UNWIND $rels AS r
MATCH (a:Entity {name: r.src, file_title: $ft})
MATCH (b:Entity {name: r.dst, file_title: $ft})
MERGE (a)-[x:REL {type: r.type, file_title: $ft}]->(b)
"""


# ========================
# Cypher：统计
# ========================
_COUNT_STATEMENTS = {
    "chunks": "MATCH (n:Chunk {file_title: $ft}) RETURN count(n) AS c",
    "entities": "MATCH (n:Entity {file_title: $ft}) RETURN count(n) AS c",
    "links": (
        "MATCH (:Entity {file_title: $ft})-[:APPEARS_IN]->(:Chunk {file_title: $ft}) "
        "RETURN count(*) AS c"
    ),
    "relations": (
        "MATCH (:Entity {file_title: $ft})-[r:REL]->(:Entity {file_title: $ft}) "
        "RETURN count(r) AS c"
    ),
}


# ========================
# Cypher：读取图谱（供前端可视化）
# ========================
# 切片不画进图：84 个切片节点会把 257 个实体淹没，只把「出现在几个切片里」挂在实体上
_READ_GRAPH_NODES = """
MATCH (e:Entity {file_title: $ft})
OPTIONAL MATCH (e)-[:APPEARS_IN]->(c:Chunk {file_title: $ft})
RETURN e.name AS name, e.type AS type, count(DISTINCT c) AS chunk_count
"""

_READ_GRAPH_EDGES = """
MATCH (a:Entity {file_title: $ft})-[r:REL]->(b:Entity {file_title: $ft})
RETURN a.name AS source, b.name AS target, r.type AS type
"""


# ========================
# Cypher：检索侧查询（node_query_kg）
# ========================
# 种子：实体名直接出现在问题文本里
_QUERY_KG_SEEDS = """
MATCH (e:Entity)
WHERE e.item_name IN $items AND $q CONTAINS e.name
RETURN DISTINCT e.name AS name
"""

# 扩展：种子的一跳邻居（REL 不写方向，入边出边都算）
_QUERY_KG_NEIGHBORS = """
MATCH (e:Entity)-[:REL]-(nb:Entity)
WHERE e.item_name IN $items AND e.name IN $seeds AND nb.item_name IN $items
RETURN DISTINCT nb.name AS name
"""

# 这些实体挂载在哪些切片上，同时带回 df（该实体出现在几个切片里）用于逆文档频率打折
_QUERY_KG_CHUNKS = """
MATCH (e:Entity)-[:APPEARS_IN]->(c:Chunk)
WHERE e.item_name IN $items AND e.name IN $names
WITH e.name AS entity, collect(DISTINCT c.chunk_id) AS chunk_ids
RETURN entity, chunk_ids, size(chunk_ids) AS df
"""

# 打分权重：直接命中的实体比一跳邻居更能代表用户意图
KG_SEED_WEIGHT = 2.0
KG_NEIGHBOR_WEIGHT = 1.0


def is_neo4j_available() -> bool:
    """
    探测 Neo4j 是否可用（导入侧的前置检查）

    图谱是补充召回，Neo4j 挂掉不该让整篇文档导入失败，所以调用方先用它探一下。
    :return: 可用返回 True
    """
    try:
        get_neo4j_driver().verify_connectivity()
        return True
    except Exception as e:
        logger.warning(f"[Neo4j] 连接不可用：{e}")
        return False


def ensure_neo4j_schema() -> bool:
    """建唯一约束与索引（幂等，全部 IF NOT EXISTS）"""
    try:
        with get_neo4j_driver().session() as session:
            for stmt in _SCHEMA_STATEMENTS:
                session.run(stmt)
        logger.info("[Neo4j] 约束与索引已就绪")
        return True
    except Exception as e:
        logger.error(f"[Neo4j] 建约束失败：{e}", exc_info=True)
        return False


def _count_doc_graph_tx(tx, file_title: str) -> Dict[str, int]:
    """在事务内统计某文档的图谱规模"""
    stats = {}
    for key, cypher in _COUNT_STATEMENTS.items():
        record = tx.run(cypher, ft=file_title).single()
        stats[key] = record["c"] if record else 0
    return stats


def _delete_doc_graph_tx(tx, file_title: str) -> Dict[str, int]:
    """在事务内清理某文档的图谱数据，返回清理前的计数便于对账"""
    before = _count_doc_graph_tx(tx, file_title)
    tx.run(_DELETE_RELATIONS, ft=file_title)
    tx.run(_DELETE_NODES, ft=file_title)
    return before


def _write_doc_graph_tx(
        tx,
        file_title: str,
        item_name: str,
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        rels: List[Dict[str, Any]],
) -> Dict[str, int]:
    """在事务内先清理后写入，最后回读实际数量"""
    # 必须先清：重复导入时 Milvus 会生成全新的 chunk_id，
    # 旧 Chunk 节点带的是失效 id，APPEARS_IN 会变成指向幽灵切片的悬空边
    tx.run(_DELETE_RELATIONS, ft=file_title)
    tx.run(_DELETE_NODES, ft=file_title)

    if chunks:
        tx.run(_WRITE_CHUNKS, chunks=chunks, ft=file_title, item=item_name)
    if entities:
        tx.run(_WRITE_ENTITIES, entities=entities, ft=file_title, item=item_name)
    if links:
        tx.run(_WRITE_LINKS, links=links, ft=file_title)
    if rels:
        tx.run(_WRITE_RELATIONS, rels=rels, ft=file_title)

    # 回读实际写入量：端点 MATCH 不到会静默丢边，调用方需要拿它对账
    return _count_doc_graph_tx(tx, file_title)


def write_doc_graph(
        file_title: str,
        item_name: str,
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        rels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    写入一份文档的图谱（先清理后写入，单事务）

    :param file_title: 文档名，本图谱的隔离与清理依据
    :param item_name: 产品主体名，作为实体的属性（不做节点）
    :param chunks: [{"chunk_id","title","parent_title"}]
    :param entities: [{"name","type"}]
    :param links: [{"name","chunk_id"}] 实体挂载到切片
    :param rels: [{"src","dst","type"}] 实体间关系
    :return: {"ok", "stats"} 或 {"ok": False, "error"}
    """
    try:
        with get_neo4j_driver().session() as session:
            stats = session.execute_write(
                _write_doc_graph_tx, file_title, item_name, chunks, entities, links, rels
            )
        logger.info(f"[Neo4j] 文档[{file_title}]图谱写入完成：{stats}")
        return {"ok": True, "stats": stats}
    except Exception as e:
        logger.error(f"[Neo4j] 文档[{file_title}]图谱写入失败：{e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def delete_doc_graph(file_title: str) -> Dict[str, Any]:
    """
    删除某文档在 Neo4j 里的全部痕迹（供文档撤回调用）

    :param file_title: 文档名
    :return: {"ok", "deleted", "detail"}
    """
    try:
        with get_neo4j_driver().session() as session:
            before = session.execute_write(_delete_doc_graph_tx, file_title)
        logger.info(f"[Neo4j] 已清理文档[{file_title}]的图谱数据：{before}")
        return {"ok": True, "deleted": sum(before.values()), "detail": before}
    except Exception as e:
        logger.error(f"[Neo4j] 清理文档[{file_title}]图谱失败：{e}", exc_info=True)
        return {"ok": False, "deleted": 0, "error": str(e)}


def count_graph_by_file_titles(file_titles: List[str]) -> Dict[str, Dict[str, int]]:
    """
    批量统计多个文档的图谱规模（供 GET /documents 展示，不用开 Neo4j Browser）

    :param file_titles: 文档名列表
    :return: {file_title: {"entities": n, "chunks": n}}；查询失败返回空字典
    """
    if not file_titles:
        return {}
    try:
        with get_neo4j_driver().session() as session:
            result = session.run(
                """
                UNWIND $titles AS ft
                OPTIONAL MATCH (e:Entity {file_title: ft})
                WITH ft, count(e) AS entities
                OPTIONAL MATCH (c:Chunk {file_title: ft})
                RETURN ft AS file_title, entities, count(c) AS chunks
                """,
                titles=file_titles,
            )
            return {
                r["file_title"]: {"entities": r["entities"], "chunks": r["chunks"]}
                for r in result
            }
    except Exception as e:
        # 图谱统计只是列表页的附加信息，查不到就不显示，不能让接口整个失败
        logger.warning(f"[Neo4j] 统计文档图谱失败，将不展示图谱信息：{e}")
        return {}


def read_doc_graph(file_title: str) -> Dict[str, Any]:
    """
    读取一份文档的实体-关系图（供前端可视化）

    切片不作为节点返回，只把「出现在几个切片里」记在实体上——
    84 个切片节点会把 257 个实体淹没，图就没法看了。

    实体 name 在 (name, file_title) 唯一约束下文档内唯一，可直接当图节点 id 使用。

    :param file_title: 文档名
    :return: {"available": bool, "nodes": [...], "edges": [...], "stats": {...}}
             available=False 表示 Neo4j 不可用或该文档没有图谱数据
    """
    empty = {"nodes": [], "edges": [], "stats": {"entities": 0, "relations": 0, "chunks": 0}}

    if not file_title:
        return {"available": False, "error": "file_title 为空", **empty}

    try:
        with get_neo4j_driver().session() as session:
            nodes = [dict(r) for r in session.run(_READ_GRAPH_NODES, ft=file_title)]
            edges = [dict(r) for r in session.run(_READ_GRAPH_EDGES, ft=file_title)]
            chunk_row = session.run(_COUNT_STATEMENTS["chunks"], ft=file_title).single()

        return {
            "available": True,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "entities": len(nodes),
                "relations": len(edges),
                # 切片总数单独查，别拿节点返回值拼
                "chunks": chunk_row["c"] if chunk_row else 0,
            },
        }
    except Exception as e:
        logger.warning(f"[Neo4j] 读取文档[{file_title}]图谱失败：{e}")
        return {"available": False, "error": str(e), **empty}


def query_kg_chunks(item_names: List[str], query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    按产品名与问题文本检索知识图谱，返回相关切片（按相关度降序）

    三步：
    1. **种子**：实体名直接出现在问题里的实体（`$q CONTAINS e.name`）
    2. **扩展**：种子的一跳邻居——这是图相对纯向量检索的价值所在
    3. **挂载**：把这些实体挂到的切片收上来，按权重累加后排序。
       权重 = 种子 2 / 邻居 1，再除以 `sqrt(该实体出现在几个切片里)` 做**逆文档频率打折**
       ——否则「烫金机」这类出现在几十个切片里的泛实体，会让含一堆泛实体的切片
       盖过真正精准的切片（实测确实会跑偏，「产品简介」压过「安装烫金膜盒」）

    按 `item_name` 跨文档查：图谱虽然按 file_title 隔离**写入**，
    但查询应按产品**聚合**（同一产品可能有多篇文档）。

    只返回 chunk_id，正文由调用方回 Milvus 取——图谱的 Chunk 节点不存 content。

    :param item_names: 已确认的产品名列表
    :param query: 改写后的问题
    :param limit: 最多返回多少个切片
    :return: [{"chunk_id": str, "score": float, "via": [实体名]}]
             无命中或 Neo4j 不可用时返回空列表（图谱是补充召回，不该让整条链路失败）
    """
    if not item_names or not query:
        return []

    try:
        with get_neo4j_driver().session() as session:
            seeds = [r["name"] for r in session.run(_QUERY_KG_SEEDS, items=item_names, q=query)]
            if not seeds:
                logger.info("[Neo4j] 问题中没有命中任何图谱实体，图谱这一路不参与")
                return []

            neighbors = [
                r["name"] for r in session.run(
                    _QUERY_KG_NEIGHBORS, items=item_names, seeds=seeds
                )
            ]
            # 种子权重高于邻居；同名以种子为准
            weight = {name: KG_SEED_WEIGHT for name in seeds}
            for name in neighbors:
                weight.setdefault(name, KG_NEIGHBOR_WEIGHT)

            rows = list(session.run(
                _QUERY_KG_CHUNKS, items=item_names, names=list(weight.keys())
            ))
    except Exception as e:
        logger.warning(f"[Neo4j] 图谱检索失败，这一路不参与：{e}")
        return []

    scores: Dict[str, float] = {}
    via: Dict[str, set] = {}
    for row in rows:
        entity = row["entity"]
        # 逆文档频率打折：像「烫金机」这种出现在几十个切片里的泛实体贡献要小，
        # 「安装烫金膜盒」这种只出现在少数切片里的具体实体贡献要大。
        # 用 1/df 而不是 1/sqrt(df)：泛实体数量多，sqrt 的折扣压不住——
        # 实测「产品简介」（靠 5 个泛实体堆分）仍会压过「安装烫金膜盒」所在的切片。
        weight_of = weight.get(entity, KG_NEIGHBOR_WEIGHT) / max(int(row["df"]), 1)
        for chunk_id in row["chunk_ids"]:
            cid = str(chunk_id)
            scores[cid] = scores.get(cid, 0.0) + weight_of
            via.setdefault(cid, set()).add(entity)

    # 分数降序；同分按 chunk_id 排，保证结果稳定
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    result = [
        {"chunk_id": chunk_id, "score": score, "via": sorted(via[chunk_id])}
        for chunk_id, score in ranked
    ]
    logger.info(
        f"[Neo4j] 图谱检索：种子实体 {len(seeds)} 个、邻居 {len(neighbors)} 个，"
        f"命中切片 {len(scores)} 个，取前 {len(result)} 个"
    )
    return result


if __name__ == '__main__':
    """
    本地测试：连通性 → 建约束 → 写入假图谱 → 校验幂等 → 清理

    前置：docker compose -f docker/neo4j-compose.yml up -d
    """
    logger.info("=" * 70)
    logger.info("[测试] 开始验证 Neo4j 图谱读写")

    TEST_FILE = "kg_util_测试文档"
    problems = []

    if not is_neo4j_available():
        logger.error("[测试] [FAIL] Neo4j 不可用，请先启动容器")
        raise SystemExit(1)
    logger.info("[测试] 连通性正常")

    if not ensure_neo4j_schema():
        raise SystemExit("[测试] [FAIL] 建约束失败")

    chunks = [
        {"chunk_id": "kgtest_1", "title": "## 装入烫金膜盒", "parent_title": "# 操作"},
        {"chunk_id": "kgtest_2", "title": "## 更换电池", "parent_title": "# 维护"},
    ]
    entities = [{"name": "烫金膜盒", "type": "部件"}, {"name": "装入烫金膜盒", "type": "操作"}]
    links = [{"name": "烫金膜盒", "chunk_id": "kgtest_1"},
             {"name": "装入烫金膜盒", "chunk_id": "kgtest_1"},
             {"name": "烫金膜盒", "chunk_id": "kgtest_2"}]
    rels = [{"src": "装入烫金膜盒", "dst": "烫金膜盒", "type": "使用"}]

    def run_once(round_name: str) -> Dict[str, int]:
        res = write_doc_graph(TEST_FILE, "测试产品", chunks, entities, links, rels)
        if not res.get("ok"):
            problems.append(f"{round_name} 写入失败：{res.get('error')}")
            return {}
        return res["stats"]

    first = run_once("第一次")
    logger.info(f"[测试] 第一次写入：{first}")

    # 幂等：同一份数据再写一次，数量不应翻倍
    second = run_once("第二次")
    logger.info(f"[测试] 第二次写入：{second}")

    if first != second:
        problems.append(f"重复写入数量不一致（幂等失败）：{first} != {second}")
    if first.get("chunks") != len(chunks):
        problems.append(f"Chunk 数不符：{first.get('chunks')} != {len(chunks)}")
    if first.get("entities") != len(entities):
        problems.append(f"Entity 数不符：{first.get('entities')} != {len(entities)}")
    if first.get("links") != len(links):
        problems.append(f"挂载边数不符：{first.get('links')} != {len(links)}")
    if first.get("relations") != len(rels):
        problems.append(f"关系数不符：{first.get('relations')} != {len(rels)}")

    # 批量统计
    counts = count_graph_by_file_titles([TEST_FILE, "不存在的文档"])
    logger.info(f"[测试] 批量统计：{counts}")
    if counts.get(TEST_FILE, {}).get("entities") != len(entities):
        problems.append("批量统计的实体数不对")

    # 读回图谱（前端可视化接口依赖它）
    g = read_doc_graph(TEST_FILE)
    logger.info(
        f"[测试] 读回图谱：available={g['available']} 节点={len(g['nodes'])} "
        f"边={len(g['edges'])} stats={g['stats']}"
    )
    if not g["available"]:
        problems.append(f"read_doc_graph 返回不可用：{g.get('error')}")
    else:
        if g["stats"]["entities"] != len(entities):
            problems.append(f"读回实体数不符：{g['stats']['entities']} != {len(entities)}")
        if g["stats"]["relations"] != len(rels):
            problems.append(f"读回关系数不符：{g['stats']['relations']} != {len(rels)}")
        if g["stats"]["chunks"] != len(chunks):
            problems.append(f"读回切片数不符：{g['stats']['chunks']} != {len(chunks)}")
        node = next((n for n in g["nodes"] if n["name"] == "烫金膜盒"), None)
        if node is None:
            problems.append("读回的节点里找不到「烫金膜盒」")
        elif node["chunk_count"] != 2:
            # 测试数据里「烫金膜盒」挂在 kgtest_1 与 kgtest_2 两个切片上
            problems.append(f"chunk_count 不符：{node['chunk_count']} != 2")
        if not any(e["source"] == "装入烫金膜盒" and e["target"] == "烫金膜盒"
                   for e in g["edges"]):
            problems.append("读回的关系里缺「装入烫金膜盒 → 烫金膜盒」")

    # 检索侧查询：种子实体 + 一跳邻居
    hits = query_kg_chunks(["测试产品"], "烫金膜盒怎么安装？")
    logger.info(f"[测试] 图谱检索命中 {len(hits)} 个切片：{[(h['chunk_id'], h['score']) for h in hits]}")
    if not hits:
        problems.append("query_kg_chunks 没有命中任何切片")
    else:
        # 测试数据里 烫金膜盒 挂 kgtest_1/kgtest_2，装入烫金膜盒 挂 kgtest_1，两者有 REL 相连；
        # 问题只含「烫金膜盒」，故它是种子、另一个是邻居。
        # 被两个实体同时命中的 kgtest_1 应排在只被种子命中的 kgtest_2 之前。
        if hits[0]["chunk_id"] != "kgtest_1":
            problems.append(f"检索排序不对：Top1 应为 kgtest_1，实际 {hits[0]['chunk_id']}")
        if not any(h["chunk_id"] == "kgtest_2" for h in hits):
            problems.append("kgtest_2 也应被召回（种子实体挂载过它）")
        via = {h["chunk_id"]: h["via"] for h in hits}
        if "装入烫金膜盒" not in via.get("kgtest_1", []):
            problems.append("kgtest_1 应经由种子与邻居两个实体命中")

    # 问题里没有实体名 / 没有产品名时，应返回空而不是报错或返回全量
    if query_kg_chunks(["测试产品"], "今天天气怎么样"):
        problems.append("问题不含实体名时不应有命中")
    if query_kg_chunks([], "烫金膜盒"):
        problems.append("item_names 为空时应返回空")

    # 清理
    del_res = delete_doc_graph(TEST_FILE)
    logger.info(f"[测试] 清理结果：{del_res}")
    if not del_res.get("ok"):
        problems.append(f"清理失败：{del_res.get('error')}")
    after = count_graph_by_file_titles([TEST_FILE]).get(TEST_FILE, {})
    if after.get("entities") or after.get("chunks"):
        problems.append(f"清理后仍有残留：{after}")

    for p in problems:
        logger.error(f"[测试] [FAIL] {p}")
    if not problems:
        logger.success("[测试] [PASS] Neo4j 图谱读写验证通过（含幂等与清理）")
    logger.info("=" * 70)
