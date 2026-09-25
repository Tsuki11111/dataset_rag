"""
文档去重记录读写工具（基于 MongoDB）

用途：记录「已导入文档的内容指纹」，上传时据此判断文件是否重复导入。

为什么需要这层（而不直接查 Milvus）：
- Milvus 里只有已成功入库的切片（kb_chunks），拿不到「处理中」「失败」这类中间状态；
  靠它无法在导入开始前拦住重复文件，也无法在失败后允许重传。
- 指纹集合是独立的「文件 → 状态」小映射，与向量检索无关，放 MongoDB 更合适。

设计沿用 app/clients/mongo_history_utils.py 的模式：工具类 + 全局单例 + 懒加载。
集合名 imported_documents，file_hash 建唯一索引，天然保证同一文件只有一条记录。
"""
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.core.logger import logger

# 集合名
COLLECTION_NAME = "imported_documents"

# 文档状态常量
STATUS_PROCESSING = "processing"   # 已受理，图正在跑
STATUS_COMPLETED = "completed"     # 导入成功
STATUS_FAILED = "failed"           # 导入失败，允许用户重传

# 去重查询只拦截这两种状态：failed 的文档允许重新导入
BLOCKING_STATUSES = [STATUS_PROCESSING, STATUS_COMPLETED]

# 计算 file_hash 的函数名，仅用于日志提示
_HASH_HELPER = "app.utils.file_hash_utils.calc_file_hash"


class DocumentDedupTool:
    """
    MongoDB 文档去重记录读写工具类

    封装连接、集合获取与索引创建，为上层提供统一的去重记录操作入口。
    """

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.collection = self.db[COLLECTION_NAME]

            # file_hash 建唯一索引：从数据库层面保证同一文件只有一条记录，
            # 也让并发上传时的「先写入者胜出」成为原子操作
            self.collection.create_index([("file_hash", 1)], unique=True)
            # 按导入时间倒序查询（列表页 / 调试用）
            self.collection.create_index([("imported_at", DESCENDING)])
            # 撤回时按 file_title 查（document_admin 用）
            self.collection.create_index([("file_title", 1)])

            logging.info(f"Successfully connected to MongoDB: {self.db_name}.{COLLECTION_NAME}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB for dedup: {e}")
            raise


# 全局单例，避免重复建立连接
_dedup_tool: Optional[DocumentDedupTool] = None

# 模块加载时预初始化，把连接建立提前，避免首次请求时才连（提升首次响应速度）
try:
    _dedup_tool = DocumentDedupTool()
except Exception as e:
    # 加载阶段失败不阻断程序启动，保留懒加载兜底
    logging.warning(f"Could not initialize DocumentDedupTool on module load: {e}")


def get_dedup_tool() -> DocumentDedupTool:
    """获取去重工具的单例实例（懒加载）"""
    global _dedup_tool
    if _dedup_tool is None:
        _dedup_tool = DocumentDedupTool()
    return _dedup_tool


def check_document_exists(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    检查文件是否已导入过

    只有 status 为 processing / completed 才算重复；failed 的允许重传
    :param file_hash: 文件SHA-256
    :return: 命中的记录字典，未命中返回 None
    """
    function_name = sys._getframe().f_code.co_name
    tool = get_dedup_tool()
    doc = tool.collection.find_one({
        "file_hash": file_hash,
        "status": {"$in": BLOCKING_STATUSES},
    })

    if doc is None:
        logger.info(f"[{function_name}] 未检测到重复文档，哈希={file_hash[:16]}...")
        return None

    doc.pop("_id", None)
    logger.warning(
        f"[{function_name}] 检测到重复文档：{doc.get('file_title')}，"
        f"状态={doc.get('status')}，导入时间={doc.get('imported_at')}"
    )
    return doc


def save_document_record(
    file_hash: str,
    file_title: str,
    status: str = STATUS_PROCESSING,
) -> bool:
    """
    写入/更新文档记录（上传受理时调用）

    先写 processing 记录，使同一文件并发上传时第二个请求能被识别为重复。
    用 upsert：强制重导时覆盖旧记录的状态与时间。
    :return: True 表示本次新建，False 表示覆盖了已有记录
    """
    function_name = sys._getframe().f_code.co_name
    tool = get_dedup_tool()
    imported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = tool.collection.update_one(
        {"file_hash": file_hash},
        {
            "$set": {
                "file_hash": file_hash,
                "file_title": file_title,
                "status": status,
                "imported_at": imported_at,
            },
            # 只在新建时置初值，覆盖时不重置已有结果
            "$setOnInsert": {"item_name": "", "chunk_count": 0},
        },
        upsert=True,
    )
    created = result.upserted_id is not None
    logger.info(f"[{function_name}] 文档记录已写入：{file_title}（{status}，{'新建' if created else '覆盖'}）")
    return created


def update_document_result(
    file_hash: str,
    status: str,
    item_name: str = "",
    chunk_count: int = 0,
) -> bool:
    """
    图执行结束后回填结果

    :param file_hash: 文件SHA-256
    :param status: completed 或 failed
    :param item_name: 识别出的产品主体名称（成功时有值）
    :param chunk_count: 入库切片数（成功时有值）
    :return: True 表示确实更新到了记录
    """
    function_name = sys._getframe().f_code.co_name
    tool = get_dedup_tool()

    result = tool.collection.update_one(
        {"file_hash": file_hash},
        {"$set": {"status": status, "item_name": item_name, "chunk_count": chunk_count}},
    )

    if result.matched_count == 0:
        # 记录不存在通常意味着记录被撤回或手动清理过，值得告警
        logger.warning(f"[{function_name}] 未找到待更新的去重记录，哈希={file_hash[:16]}...")
        return False

    if status == STATUS_COMPLETED:
        logger.info(f"[{function_name}] 文档导入完成：{item_name}，切片数={chunk_count}")
    else:
        logger.error(f"[{function_name}] 文档导入失败，已标记为failed便于重新上传：{file_hash[:16]}...")
    return True


def list_documents(limit: int = 50) -> List[Dict[str, Any]]:
    """
    列出已导入文档（便于查看/调试）
    :param limit: 返回条数上限
    :return: 记录字典列表，按导入时间倒序
    """
    tool = get_dedup_tool()
    cursor = tool.collection.find({}).sort("imported_at", DESCENDING).limit(limit)
    rows = []
    for doc in cursor:
        doc.pop("_id", None)
        rows.append(doc)
    return rows


def get_records_by_titles(file_titles: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    按 file_title 批量取记录，返回 {file_title: 记录} 映射

    供文档列表聚合使用：Milvus 是文档列表的主来源，这里补充导入时间等信息
    :param file_titles: 文档名列表（去扩展名）
    """
    if not file_titles:
        return {}
    tool = get_dedup_tool()
    cursor = tool.collection.find(
        {"file_title": {"$in": file_titles}},
        {"file_title": 1, "item_name": 1, "imported_at": 1, "status": 1, "file_hash": 1},
    )
    result = {}
    for doc in cursor:
        doc.pop("_id", None)
        result[doc["file_title"]] = doc
    return result


def clear_by_file_title(file_title: str) -> int:
    """
    按文档名删除去重记录（撤回文档时调用）

    删除后该文件可以重新上传。按 file_title 删而非 file_hash：
    撤回时只知道文档名，拿不到原始文件的哈希。
    :return: 实际删除条数
    """
    function_name = sys._getframe().f_code.co_name
    tool = get_dedup_tool()
    result = tool.collection.delete_many({"file_title": file_title})
    logger.info(f"[{function_name}] 已删除文档[{file_title}]的去重记录 {result.deleted_count} 条")
    return result.deleted_count


def clear_all() -> int:
    """清空全部去重记录（测试用）"""
    tool = get_dedup_tool()
    result = tool.collection.delete_many({})
    return result.deleted_count


if __name__ == '__main__':
    """本地测试：打印现有去重记录"""
    logger.info(f"去重集合：{os.getenv('MONGO_DB_NAME')}.{COLLECTION_NAME}")
    records = list_documents()
    logger.info(f"当前已导入文档记录数：{len(records)}")
    for r in records:
        logger.info(
            f"  {r.get('file_title')} | 产品={r.get('item_name')} | 切片={r.get('chunk_count')} | "
            f"{r.get('status')} | {r.get('imported_at')}"
        )
