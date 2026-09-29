"""
文件导入 Web 服务

把 LangGraph 知识库导入流程（PDF/MD → 解析 → 切分 → 向量化 → Milvus入库）
包装成 HTTP 接口，并提供可视化上传页面。

核心特性：
1. 文件上传后、跑图前，先按文件内容 SHA-256 校验是否已导入过（重复则跳过）
2. 支持 force=true 强制重新导入
3. 后台任务执行导入图，前端轮询 /status 获取节点级进度
"""
import os
import shutil
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.clients.minio_utils import get_minio_client
from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.clients.mongo_dedup_utils import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    check_document_exists,
    save_document_record,
    update_document_result,
)
from app.utils.file_hash_utils import calc_file_hash
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)

# 允许上传的文件后缀
ALLOWED_EXTENSIONS = {".pdf", ".md"}
# 单文件大小上限（200MB），防止异常大文件
MAX_FILE_SIZE = 200 * 1024 * 1024

app = FastAPI(
    title="File Import Service",
    description="知识库导入服务：上传 PDF/MD → 解析 → 切分 → 向量化 → Milvus入库（含重复文档检测）"
)

# 跨域配置：允许前端页面独立部署时调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    """
    服务启动时预热去重工具

    MongoDB 的集合与索引在 DocumentDedupTool 实例化时自动创建（幂等），
    这里主动取一次单例，把连接建立提前到启动阶段，避免首次上传时才连。
    """
    from app.clients.mongo_dedup_utils import get_dedup_tool
    get_dedup_tool()
    logger.info("去重工具已就绪（MongoDB）")
    logger.info("File Import Service 启动完成")


@app.get("/import.html", response_class=FileResponse, summary="文件上传页面")
async def get_import_page():
    """返回前端上传页面"""
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")
    return FileResponse(path=html_abs_path, media_type="text/html")


def run_graph_task(task_id: str, local_dir: str, local_file_path: str, file_hash: str):
    """
    LangGraph 全流程后台任务

    由 BackgroundTasks 触发，不阻塞 HTTP 响应。
    逐节点流式执行图，每完成一个节点就更新任务进度，供前端轮询。
    执行结束后把结果（产品名、切片数）回填到 SQLite 去重记录。

    :param task_id: 任务唯一ID
    :param local_dir: 该任务的本地工作目录
    :param local_file_path: 上传文件的本地绝对路径
    :param file_hash: 文件SHA-256，用于回填去重记录
    """
    function_name = sys._getframe().f_code.co_name
    update_task_status(task_id, "processing")
    logger.info(f"[{task_id}] 开始执行LangGraph全流程，文件：{local_file_path}")

    try:
        # 构造图初始状态：只需 task_id / local_file_path / local_dir
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        # 流式执行：每完成一个节点就记录，前端轮询可见进度
        final_state: Dict[str, Any] = {}
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                logger.info(f"[{task_id}] 节点执行完成：{node_name}")
                add_done_task(task_id, node_name)
                if isinstance(node_result, dict):
                    final_state.update(node_result)

        # 回填导入结果到去重记录
        chunks = final_state.get("chunks") or []
        update_document_result(
            file_hash=file_hash,
            status=STATUS_COMPLETED,
            item_name=final_state.get("item_name") or "",
            chunk_count=len(chunks),
        )
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] 全流程执行完毕，入库切片数：{len(chunks)}")

    except Exception as e:
        # 标记 failed：让用户能重新上传同一文件（去重只拦截 processing/completed）
        update_document_result(file_hash=file_hash, status=STATUS_FAILED)
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] 全流程执行失败：{str(e)}", exc_info=True)


def _save_upload_file(file: UploadFile, dest_path: str) -> None:
    """
    把上传文件分块写入磁盘（避免大文件占满内存）
    :param file: FastAPI的UploadFile对象
    :param dest_path: 目标绝对路径
    """
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer, length=1024 * 1024)


@app.post("/upload", summary="文件上传接口")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="待导入的文件（PDF/MD），支持多选"),
    force: bool = Form(False, description="true表示即使检测到重复也强制重新导入"),
):
    """
    文件上传接口（含重复文档检测）

    流程：接收文件 → 存本地并计算哈希 → 查重 → 未重复才启动图

    :return: task_ids（已受理的任务）、duplicates（被跳过的重复文件）
    """
    function_name = sys._getframe().f_code.co_name
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    task_ids: List[str] = []
    duplicates: List[Dict[str, Any]] = []
    failed_files: List[Dict[str, str]] = []

    logger.info(f"[{function_name}] 收到上传请求，文件数={len(files)}，force={force}")

    for file in files:
        original_name = file.filename or "unnamed"
        ext = os.path.splitext(original_name)[1].lower()

        # 1. 后缀校验
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"[{function_name}] 跳过不支持的文件类型：{original_name}（{ext}）")
            failed_files.append({"filename": original_name, "reason": f"仅支持 {ALLOWED_EXTENSIONS}，当前为 {ext}"})
            continue

        # 2. 先落到临时目录
        task_id = str(uuid.uuid4())
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)
        local_file_abs_path = os.path.join(task_local_dir, original_name)
        try:
            _save_upload_file(file, local_file_abs_path)
        except Exception as e:
            logger.error(f"[{function_name}] 文件保存失败：{original_name}，{e}", exc_info=True)
            failed_files.append({"filename": original_name, "reason": f"文件保存失败：{e}"})
            continue

        # 3. 大小校验
        file_size = os.path.getsize(local_file_abs_path)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"[{function_name}] 文件过大：{original_name}（{file_size}字节）")
            failed_files.append({"filename": original_name, "reason": f"文件超过{MAX_FILE_SIZE // 1024 // 1024}MB上限"})
            shutil.rmtree(task_local_dir, ignore_errors=True)
            continue

        # 4. 计算内容哈希并查重
        file_hash = calc_file_hash(local_file_abs_path)
        file_title = os.path.splitext(original_name)[0]
        existing = check_document_exists(file_hash)

        if existing and not force:
            # 命中重复且未强制：清掉刚落的临时文件，记入 duplicates 返回给前端
            logger.warning(f"[{function_name}] 文件已导入过，跳过：{original_name}")
            duplicates.append({
                "filename": original_name,
                "file_title": existing.get("file_title", ""),
                "item_name": existing.get("item_name", ""),
                "chunk_count": existing.get("chunk_count", 0),
                "status": existing.get("status", ""),
                "imported_at": existing.get("imported_at", ""),
            })
            shutil.rmtree(task_local_dir, ignore_errors=True)
            continue

        if existing and force:
            logger.warning(f"[{function_name}] 文件已导入过，但force=true，将重新导入：{original_name}")

        # 5. 标记上传阶段
        add_running_task(task_id, "upload_file")

        # 6. 上传原始文件到 MinIO 做持久化（失败不中断，本地文件仍可处理）
        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{original_name}"
        try:
            minio_client = get_minio_client()
            if minio_client is None:
                raise RuntimeError("MinIO客户端不可用")
            minio_client.fput_object(
                bucket_name=os.getenv("MINIO_BUCKET_NAME", "knowledge-base-files"),
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type,
            )
            logger.info(f"[{task_id}] 文件已上传MinIO：{minio_object_name}")
        except Exception as e:
            logger.warning(f"[{task_id}] 文件上传MinIO失败，继续本地处理：{str(e)}")

        add_done_task(task_id, "upload_file")

        # 7. 先写去重记录（processing），使并发上传同一文件时第二个请求被判为重复
        save_document_record(file_hash, file_title, status=STATUS_PROCESSING)

        # 8. 启动后台导入任务
        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path, file_hash)
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 已加入后台任务队列，文件名：{original_name}")

    return {
        "code": 200,
        "message": f"受理 {len(task_ids)} 个文件，跳过 {len(duplicates)} 个重复文件",
        "task_ids": task_ids,
        "duplicates": duplicates,
        "failed": failed_files,
    }


@app.get("/status/{task_id}", summary="任务状态查询")
async def get_task_progress(task_id: str):
    """
    查询单个任务的处理进度（前端每2秒轮询）
    数据来自内存态任务字典，无IO开销
    """
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }


@app.get("/documents", summary="已导入文档列表")
async def list_imported_documents():
    """
    列出已导入的文档（以 Milvus 为准聚合，历史文档也能列出）

    注意：不能只读 SQLite 去重记录，因为命令行导入的文档没有记录。
    """
    from app.utils.document_admin import list_imported_documents as aggregate_documents
    documents = aggregate_documents()
    return {"code": 200, "total": len(documents), "documents": documents}


@app.get("/documents/graph", summary="查询单文档知识图谱")
async def get_document_graph(file_title: str):
    """
    读取一份文档在 Neo4j 里的实体-关系图，供前端可视化

    用查询参数而非路径参数：中文文档名会自动 percent-decode，
    也不会与 DELETE /documents/{file_title} 的路由语义纠缠。

    统一返回 HTTP 200 并带 available 字段——Neo4j 不可用或该文档没有图谱时
    available=false，前端只需判这一个字段，不必处理非 2xx。
    """
    function_name = sys._getframe().f_code.co_name
    from app.clients.neo4j_utils import read_doc_graph

    logger.info(f"[{function_name}] 查询图谱：{file_title}")
    result = read_doc_graph(file_title)

    if not result["available"]:
        return {
            "code": 503,
            "available": False,
            "file_title": file_title,
            "message": "Neo4j 不可用，或该文档还没有图谱数据",
            "nodes": [],
            "edges": [],
            "stats": result["stats"],
        }

    return {
        "code": 200,
        "available": True,
        "file_title": file_title,
        "nodes": result["nodes"],
        "edges": result["edges"],
        "stats": result["stats"],
    }


@app.delete("/documents/{file_title}", summary="撤回已导入文档")
async def revoke_document_api(file_title: str, confirm: bool = False):
    """
    撤回一份已导入文档：删除 Milvus 切片与产品名、SQLite 去重记录、本地产物、MinIO 对象

    删除不可逆。默认 confirm=false，只返回将被删除的内容清单供前端二次确认；
    显式传 confirm=true 才真正执行。
    """
    function_name = sys._getframe().f_code.co_name
    from app.utils.document_admin import locate_local_artifacts, revoke_document

    targets = locate_local_artifacts(file_title)

    # 未确认：只预览将要删除的内容，不执行
    if not confirm:
        logger.info(f"[{function_name}] 撤回预览（未执行）：{file_title}")
        return {
            "code": 200,
            "confirmed": False,
            "file_title": file_title,
            "preview": {
                "local_paths": targets,
                "note": "以上为本地产物；Milvus 与 MinIO 中的相关数据也将在确认后一并删除",
            },
        }

    # 已确认：执行撤回
    result = revoke_document(file_title)
    if not result.get("success"):
        logger.error(f"[{function_name}] 撤回存在失败项：{result}")
    return {"code": 200, "confirmed": True, "result": result}


if __name__ == "__main__":
    """
    服务启动入口
    注意：8000端口已被Attu（Milvus可视化）占用，本服务使用8001
    """
    logger.info("File Import Service 服务启动中（端口8001）...")
    uvicorn.run(app=app, host="127.0.0.1", port=8001)
