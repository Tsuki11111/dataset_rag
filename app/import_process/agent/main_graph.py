from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.core.logger import logger
from app.core.usage_tracker import add_tracked_node, usage_context
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_dashscope_embedding import node_dashscope_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_kg import node_import_kg
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState, create_default_state

load_dotenv()

# 初始化langgraph状态图
workflow = StateGraph(ImportGraphState)
# 注册所有子节点
# 用 add_tracked_node 包一层归因：节点内的模型调用记到该节点名下，见 app/core/usage_tracker.py
add_tracked_node(workflow, "node_entry", node_entry)
add_tracked_node(workflow, "node_pdf_to_md", node_pdf_to_md)
add_tracked_node(workflow, "node_md_img", node_md_img)
add_tracked_node(workflow, "node_document_split", node_document_split)
add_tracked_node(workflow, "node_item_name_recognition", node_item_name_recognition)
add_tracked_node(workflow, "node_dashscope_embedding", node_dashscope_embedding)
add_tracked_node(workflow, "node_import_milvus", node_import_milvus)
# 图谱构建依赖 Milvus 回填的 chunk_id，所以必须排在 node_import_milvus 之后
add_tracked_node(workflow, "node_import_kg", node_import_kg)

# 设置入口节点
workflow.set_entry_point("node_entry")
# 定义条件边的路由函数
def route_after_entry(state: ImportGraphState) -> str:
    """
    文件是pdf->node_pdf_to_md
    文件是md->node_md_img
    既不是文件又不是markdown->END
    :param state:
    :return:
    """
    if state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    elif state["is_md_read_enabled"]:
        return "node_md_img"
    else:
        return END

workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    path_map={
        "node_pdf_to_md": "node_pdf_to_md",
        "node_md_img": "node_md_img",
        END: END,
    }
)
# 定义静态边
# PDF 路径也要经过 node_md_img：MinerU 产出的是本地相对路径 ![](images/xxx.jpg)，
# 需要由 node_md_img 上传到 MinIO 并用视觉模型生成图片描述，才能让检索拿到可访问的图片URL
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_dashscope_embedding")
workflow.add_edge("node_dashscope_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", "node_import_kg")
workflow.add_edge("node_import_kg", END)
# 编译节点
kb_import_app = workflow.compile()


if __name__ == '__main__':
    """
    端到端测试：从 node_entry 开始跑完整导入链路
    node_entry → node_pdf_to_md → node_md_img → node_document_split
               → node_item_name_recognition → node_dashscope_embedding → node_import_milvus

    前置条件：
    1. Docker 服务已启动：minio、milvus-etcd、milvus-standalone
    2. 已执行 create_collections.py 建库（kb_item_names / kb_chunks）
    3. .env 配置好 MINERU_API_TOKEN（PDF解析走MinerU云端）、FUNCTION
    说明：Neo4j（实体抽取）不在本图中，不受影响
    """
    import os
    import re
    import time

    from app.clients.milvus_utils import get_milvus_client
    from app.conf.milvus_config import milvus_config
    from app.utils.path_util import PROJECT_ROOT

    # --- 测试配置 ---
    # 待处理的PDF，换成 doc/ 下任意一个文件即可
    TEST_PDF_NAME = "hl3040网络说明书.pdf"

    test_pdf_path = os.path.join(PROJECT_ROOT, "doc", TEST_PDF_NAME)
    test_task_id = f"graph_test_{int(time.time())}"

    logger.info("=" * 70)
    logger.info(f"[main_graph测试] 开始端到端测试，task_id={test_task_id}")
    logger.info(f"[main_graph测试] 测试文档：{test_pdf_path}")
    logger.info("=" * 70)

    # --- 前置校验 ---
    if not os.path.exists(test_pdf_path):
        logger.error(f"[main_graph测试] 测试PDF不存在：{test_pdf_path}")
        raise FileNotFoundError(test_pdf_path)

    client = get_milvus_client()
    if client is None:
        logger.error("[main_graph测试] Milvus不可用，请先启动 docker start minio milvus-etcd milvus-standalone")
        raise RuntimeError("Milvus不可用")
    for col in (milvus_config.item_name_collection, milvus_config.chunks_collection):
        if not client.has_collection(col):
            logger.error(f"[main_graph测试] 集合[{col}]不存在，请先执行 create_collections.py 建库")
            raise RuntimeError(f"集合[{col}]不存在")

    # --- 构造初始状态（只有入口节点需要的两个字段）---
    # 中间产物由各节点写到 output/<文档名>/ 下（如 md、chunks.json、images），
    # 天然按文档隔离，因此这里用公共的 output/ 即可
    init_state = create_default_state(
        task_id=test_task_id,
        local_file_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )

    # --- 执行图 ---
    logger.info("[main_graph测试] 开始执行图...")
    start = time.time()
    try:
        # 包一层记账/日志上下文：命令行跑图也要有一条 trace，
        # 否则日志只有节点名没有 trace、账目也归不到「哪一次运行」（服务入口是包了的）
        with usage_context(session_id=test_task_id) as acc:
            final_state = kb_import_app.invoke(init_state)
    except Exception as e:
        logger.error(f"[main_graph测试] 图执行失败：{str(e)}", exc_info=True)
        raise
    elapsed = time.time() - start
    logger.info(f"[main_graph测试] 本次导入记账：{acc.text()}")

    # --- 结果校验 ---
    logger.info("=" * 70)
    logger.info(f"[main_graph测试] 图执行完成，耗时 {elapsed:.1f} 秒")
    logger.info("=" * 70)

    chunks = final_state.get("chunks") or []
    item_name = final_state.get("item_name") or ""
    has_vector = sum(1 for c in chunks if c.get("dense_vector"))
    has_chunk_id = sum(1 for c in chunks if c.get("chunk_id"))

    logger.info(f"[main_graph测试] 路由: is_pdf_read_enabled={final_state.get('is_pdf_read_enabled')}")
    logger.info(f"[main_graph测试] md_path: {final_state.get('md_path')}")
    logger.info(f"[main_graph测试] 切分切片数: {len(chunks)}")
    logger.info(f"[main_graph测试] 识别产品名: {item_name}")
    logger.info(f"[main_graph测试] 已生成向量的切片数: {has_vector}")
    logger.info(f"[main_graph测试] 已回填chunk_id的切片数: {has_chunk_id}")
    logger.info(f"[main_graph测试] 产品名已存kb_item_names: {final_state.get('item_name_saved')}")

    # 图片处理校验：node_md_img 应把本地图片上传MinIO并把md里的相对路径换成可访问URL
    md_content = final_state.get("md_content") or ""
    local_img_refs = re.findall(r"!\[[^\]]*\]\((?!https?://)[^)]+\)", md_content)
    minio_img_refs = re.findall(r"!\[[^\]]*\]\((https?://[^)]+)\)", md_content)
    logger.info(f"[main_graph测试] md中本地图片引用(应为0): {len(local_img_refs)}")
    logger.info(f"[main_graph测试] md中MinIO图片引用: {len(minio_img_refs)}")
    if minio_img_refs:
        logger.info(f"[main_graph测试] 图片URL样例: {minio_img_refs[0][:110]}")

    # 回查Milvus，确认真的落库了
    db_chunks = client.query(
        collection_name=milvus_config.chunks_collection,
        filter=f'file_title != ""',
        output_fields=["chunk_id", "title", "item_name"],
        limit=5,
    )
    db_item_names = client.query(
        collection_name=milvus_config.item_name_collection,
        filter=f'item_name == "{item_name}"',
        output_fields=["item_name", "file_title"],
    ) if item_name else []

    logger.info(f"[main_graph测试] kb_chunks 回查样例({len(db_chunks)}条)：")
    for row in db_chunks:
        logger.info(f"[main_graph测试]    chunk_id={row.get('chunk_id')} title={str(row.get('title'))[:30]}")

    # 断言：任一环节没产出数据就明确失败，避免"跑完了但什么都没入库"
    problems = []
    if not final_state.get("md_path"):
        problems.append("PDF未转出Markdown")
    if not chunks:
        problems.append("未切分出任何切片")
    if not item_name:
        problems.append("未识别出产品名")
    if has_vector != len(chunks):
        problems.append(f"仅{has_vector}/{len(chunks)}个切片生成了向量")
    if has_chunk_id != len(chunks):
        problems.append(f"仅{has_chunk_id}/{len(chunks)}个切片回填了chunk_id")
    if not db_item_names:
        problems.append(f"kb_item_names 中查不到产品名[{item_name}]")
    if local_img_refs:
        problems.append(f"md中仍残留{len(local_img_refs)}处本地图片引用，node_md_img未生效")
    if minio_img_refs and not all(u.startswith("http") for u in minio_img_refs):
        problems.append("存在非HTTP的图片URL")

    logger.info("=" * 70)
    if problems:
        for p in problems:
            logger.error(f"[main_graph测试] [FAIL] {p}")
        logger.error(f"[main_graph测试] 测试失败，共{len(problems)}项未通过")
    else:
        logger.success(f"[main_graph测试] [PASS] 端到端测试全部通过，共入库{len(chunks)}个切片")
    logger.info("=" * 70)
