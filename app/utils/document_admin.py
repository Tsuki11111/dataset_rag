"""
文档管理工具：列出已导入文档、定位其产物、执行撤回

撤回一份文档需要清理五处数据：
1. Milvus kb_chunks    —— 该文档的全部切片
2. Milvus kb_item_names —— 该文档的产品主体名
3. MongoDB imported_documents —— 去重记录（删除后该文件可重新上传）
4. 本地 output 目录 + MinIO 对象
5. Neo4j 知识图谱 —— 该文档的实体、切片与关系

安全原则：
- 本地路径只做「精确名匹配」，不做模糊匹配，避免误删
- Milvus / Neo4j 一律按 file_title 过滤，绝不用 item_name：
  item_name 是主键或 LLM 输出，若另一文档识别出同名产品会覆盖该行（file_title 随之改变），
  按 item_name 删会把已经属于别的文档的记录删掉
- 逐项独立 try/except，单点失败不中断整体，最后汇总报告
"""
import os
import shutil
import sys
from collections import Counter
from typing import Any, Dict, List

from minio.deleteobjects import DeleteObject

from app.clients.milvus_utils import get_milvus_client
from app.clients.minio_utils import get_minio_client
from app.clients.mongo_dedup_utils import clear_by_file_title, get_records_by_titles
from app.clients.neo4j_utils import count_graph_by_file_titles, delete_doc_graph
from app.conf.minio_config import minio_config
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.path_util import PROJECT_ROOT

# 本地输出根目录
OUTPUT_DIR = PROJECT_ROOT / "output"
# MinIO 中图片与原始文件的对象前缀（不带前导斜杠，实测带斜杠会匹配到 0 个对象）
MINIO_IMAGE_PREFIX = minio_config.minio_img_dir.lstrip("/")
MINIO_PDF_PREFIX = os.getenv("MINIO_PDF_DIR", "pdf_files")


def list_imported_documents() -> List[Dict[str, Any]]:
    """
    列出所有已导入文档

    以 Milvus kb_chunks 为主聚合：MongoDB 只记录走过 Web 上传的文档，
    命令行/早期导入的文档没有记录，只读 MongoDB 会漏掉它们。
    MongoDB 记录用于补充 item_name 和 imported_at（有则更准确）。

    :return: 文档信息列表，按切片数倒序
    """
    function_name = sys._getframe().f_code.co_name
    client = get_milvus_client()
    if client is None:
        logger.error(f"[{function_name}] Milvus不可用，无法列出文档")
        return []

    # 1. 从 kb_chunks 聚合：file_title -> 切片数
    rows = client.query(
        collection_name=milvus_config.chunks_collection,
        filter='chunk_id >= 0',
        output_fields=["file_title"],
    )
    chunk_counts = Counter(r["file_title"] for r in rows if r.get("file_title"))

    # 2. 从 kb_item_names 取每个文档的产品名（按 file_title 对应）
    item_rows = client.query(
        collection_name=milvus_config.item_name_collection,
        filter='item_name != ""',
        output_fields=["item_name", "file_title"],
    )
    item_by_title = {
        r["file_title"]: r["item_name"]
        for r in item_rows
        if r.get("file_title")
    }

    # 3. MongoDB 去重记录补充导入时间（不是所有文档都有记录，命令行导入的就没有）
    mongo_info: Dict[str, Dict[str, Any]] = {}
    try:
        mongo_info = get_records_by_titles(list(chunk_counts.keys()))
    except Exception as e:
        logger.warning(f"[{function_name}] 读取Mongo去重记录失败（不影响列表）：{e}")

    # 4. Neo4j 图谱规模。Neo4j 不可用时返回空字典，列表照常展示——
    #    否则图谱一挂，连文档列表都用不了
    kg_info: Dict[str, Dict[str, int]] = {}
    try:
        kg_info = count_graph_by_file_titles(list(chunk_counts.keys()))
    except Exception as e:
        logger.warning(f"[{function_name}] 读取Neo4j图谱统计失败（不影响列表）：{e}")

    # 5. 合并
    documents = []
    for file_title, chunk_count in chunk_counts.items():
        extra = mongo_info.get(file_title, {})
        documents.append({
            "file_title": file_title,
            # 产品名优先用 Milvus 的（它是实际入库的），去重记录兜底
            "item_name": item_by_title.get(file_title) or extra.get("item_name", ""),
            "chunk_count": chunk_count,
            "imported_at": extra.get("imported_at", ""),
            "status": extra.get("status", ""),
            # 图谱实体数；Neo4j 不可用时为 0
            "kg_entities": kg_info.get(file_title, {}).get("entities", 0),
        })

    documents.sort(key=lambda d: d["chunk_count"], reverse=True)
    logger.info(f"[{function_name}] 共列出{len(documents)}份已导入文档")
    return documents


def locate_local_artifacts(file_title: str) -> List[str]:
    """
    定位某文档在本地的全部产物路径（只查找，不删除）

    存在三种布局，都需覆盖：
    1. 命令行导入：output/<file_title>/ 与 output/<file_title>_result.zip
    2. Web 上传 PDF：output/<YYYYMMDD>/<task_id>/<file_title>/ （经 node_pdf_to_md 解压）
    3. Web 上传 MD：  output/<YYYYMMDD>/<task_id>/<file_title>.md + 同目录的 chunks.json
       —— MD 不走 node_pdf_to_md，文件由上传接口直接落盘在 task 目录下，没有同名子目录

    只做**精确名匹配**（目录名/文件名严格等于 file_title），不做模糊匹配。
    :param file_title: 文档名（去扩展名）
    :return: 待删除的绝对路径列表
    """
    function_name = sys._getframe().f_code.co_name
    targets: List[str] = []

    if not file_title or not OUTPUT_DIR.exists():
        return targets

    # 1. 命令行导入的布局：直接在 output 下
    direct_dir = OUTPUT_DIR / file_title
    if direct_dir.is_dir():
        targets.append(str(direct_dir))
    direct_zip = OUTPUT_DIR / f"{file_title}_result.zip"
    if direct_zip.is_file():
        targets.append(str(direct_zip))

    # 2/3. Web 上传的布局：output/<YYYYMMDD>/<task_id>/ 下
    for date_dir in OUTPUT_DIR.iterdir():
        if not date_dir.is_dir():
            continue
        for task_dir in date_dir.iterdir():
            if not task_dir.is_dir():
                continue

            # 形态2：PDF 解压出的同名子目录
            sub_dir = task_dir / file_title
            if sub_dir.is_dir():
                targets.append(str(sub_dir))

            # 形态3：MD 直接落盘的文件（及可能存在的 _new.md）
            for suffix in (".md", "_new.md"):
                f = task_dir / f"{file_title}{suffix}"
                if f.is_file():
                    targets.append(str(f))

            # MD 形态下 chunks.json 与 md 同目录；删掉 md 后它就成了孤儿，一并清理
            # 仅当该 task 目录确实属于本文档时才删，避免误删其他文档的备份
            if (task_dir / f"{file_title}.md").is_file() or (task_dir / file_title).is_dir():
                ck = task_dir / "chunks.json"
                if ck.is_file():
                    targets.append(str(ck))

            # 形态2 可能带 zip
            zip_file = task_dir / f"{file_title}_result.zip"
            if zip_file.is_file():
                targets.append(str(zip_file))

    logger.info(f"[{function_name}] 文档[{file_title}]定位到{len(targets)}个本地产物")
    return targets


def _revoke_milvus(file_title: str) -> Dict[str, Any]:
    """删除文档在 Milvus 两个集合中的数据（按 file_title，非 item_name）"""
    client = get_milvus_client()
    if client is None:
        return {"ok": False, "error": "Milvus不可用"}

    safe_title = escape_milvus_string(file_title)
    expr = f'file_title == "{safe_title}"'
    result: Dict[str, Any] = {}

    for label, collection in (
        ("chunks", milvus_config.chunks_collection),
        ("item_names", milvus_config.item_name_collection),
    ):
        try:
            before = client.query(collection_name=collection, filter=expr, output_fields=["file_title"])
            client.delete(collection_name=collection, filter=expr)
            # 必须flush，否则删除不会立即生效
            client.flush(collection)
            result[label] = len(before)
        except Exception as e:
            logger.error(f"删除 Milvus {collection} 失败：{e}", exc_info=True)
            result[label] = f"失败：{e}"
    return {"ok": True, "detail": result}


def _revoke_neo4j(file_title: str) -> Dict[str, Any]:
    """
    删除文档在 Neo4j 知识图谱里的数据

    安全性依据同 Milvus：实体与切片节点都带 file_title、不存在跨文档共享节点，
    所以 DETACH DELETE 只会带走本文档的节点与它自己的边。
    Neo4j 不可用时返回 ok=False，但不影响其余撤回步骤。
    """
    return delete_doc_graph(file_title)


def _revoke_dedup_record(file_title: str) -> Dict[str, Any]:
    """
    删除去重记录（删除后该文件可重新上传）

    按 file_title 删而非 file_hash：撤回时只知道文档名，拿不到原始文件哈希
    """
    try:
        deleted = clear_by_file_title(file_title)
        return {"ok": True, "deleted": deleted}
    except Exception as e:
        logger.error(f"删除去重记录失败：{e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def _revoke_local(paths: List[str]) -> Dict[str, Any]:
    """
    删除本地输出目录与zip包

    删除后再做一次「空壳清理」：Web 上传的产物位于 output/<日期>/<task_id>/ 下，
    撤回只删该文档的文件，task 目录本身会变空。若整个 task 目录已空则一并删除，
    避免留下垃圾目录；若目录里还有别的东西则保留（可能有其他文档的产物）。
    """
    deleted, errors = 0, []
    parent_dirs = set()

    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
                # 记录文件所属目录，稍后判断是否变空
                parent_dirs.add(os.path.dirname(p))
            deleted += 1
        except Exception as e:
            logger.error(f"删除本地路径失败：{p}，{e}")
            errors.append(f"{p}：{e}")

    # 空壳清理：只删「变空」的 task 目录；其父级日期目录也顺带清理（若也空了）
    for d in parent_dirs:
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                logger.info(f"清理空目录：{d}")
                date_dir = os.path.dirname(d)
                # 日期目录（如 output/20260919）下若无 task 目录了，也删掉
                if os.path.basename(date_dir).isdigit() and os.path.isdir(date_dir) and not os.listdir(date_dir):
                    os.rmdir(date_dir)
                    logger.info(f"清理空日期目录：{date_dir}")
        except Exception as e:
            logger.warning(f"清理空目录失败（不影响撤回结果）：{d}，{e}")

    return {"ok": not errors, "deleted": deleted, "errors": errors}


def _revoke_minio(file_title: str) -> Dict[str, Any]:
    """
    删除 MinIO 中该文档的图片与原始文件

    图片对象名前缀固定为 upload-images/<file_title>/
    原始文件对象名形如 pdf_files/<日期>/<文件名>，只能按文件名精确匹配后删除
    """
    mc = get_minio_client()
    if mc is None:
        return {"ok": False, "error": "MinIO不可用"}
    bucket = minio_config.bucket_name

    try:
        to_delete: List[str] = []

        # 1. 图片：按 file_title 前缀批量匹配
        for obj in mc.list_objects(bucket, prefix=f"{MINIO_IMAGE_PREFIX}/{file_title}/", recursive=True):
            to_delete.append(obj.object_name)

        # 2. 原始文件：pdf_files/<任意日期>/<file_title>.<任意扩展名>
        #    对象名含原始文件名（file_title + 扩展名），日期未知，故逐层列出后精确比对
        for obj in mc.list_objects(bucket, prefix=f"{MINIO_PDF_PREFIX}/", recursive=True):
            name = obj.object_name.rsplit("/", 1)[-1]
            if os.path.splitext(name)[0] == file_title:
                to_delete.append(obj.object_name)

        if not to_delete:
            return {"ok": True, "deleted": 0}

        errors = list(mc.remove_objects(bucket, [DeleteObject(o) for o in to_delete]))
        for e in errors:
            logger.error(f"MinIO删除失败：{e}")
        return {"ok": not errors, "deleted": len(to_delete) - len(errors), "errors": [str(e) for e in errors]}
    except Exception as e:
        logger.error(f"清理MinIO失败：{e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def revoke_document(file_title: str) -> Dict[str, Any]:
    """
    撤回一份已导入文档：删除它在 Milvus、SQLite、output、MinIO 的全部痕迹

    四处独立执行，任一处失败不影响其余部分，最后汇总报告。
    :param file_title: 文档名（去扩展名）
    :return: 各处删除结果汇总
    """
    function_name = sys._getframe().f_code.co_name
    logger.warning(f"[{function_name}] 开始撤回文档：{file_title}")

    local_paths = locate_local_artifacts(file_title)
    report: Dict[str, Any] = {
        "file_title": file_title,
        "local_paths": local_paths,
        "milvus": _revoke_milvus(file_title),
        "neo4j": _revoke_neo4j(file_title),
        "dedup": _revoke_dedup_record(file_title),
        "local": _revoke_local(local_paths),
        "minio": _revoke_minio(file_title),
    }

    # 汇总：任何一处 ok=False 都算部分失败
    all_ok = all(
        report[k].get("ok", False)
        for k in ("milvus", "neo4j", "dedup", "local", "minio")
    )
    report["success"] = all_ok
    if all_ok:
        logger.success(f"[{function_name}] 文档[{file_title}]已完整撤回")
    else:
        logger.error(f"[{function_name}] 文档[{file_title}]撤回存在失败项：{report}")
    return report


if __name__ == '__main__':
    """本地测试：查看已导入文档列表"""
    for doc in list_imported_documents():
        logger.info(
            f"  {doc['file_title']} | 产品={doc['item_name']} | 切片={doc['chunk_count']} | "
            f"导入时间={doc['imported_at'] or '(无记录)'}"
        )
        for p in locate_local_artifacts(doc["file_title"]):
            logger.info(f"      本地产物：{p}")
