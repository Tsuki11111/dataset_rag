"""
文件哈希工具

只负责计算文件内容指纹（SHA-256）。
去重记录的读写已迁至 MongoDB，见 app/clients/mongo_dedup_utils.py。
"""
import hashlib
import os
import sys

from app.core.logger import logger

# 分块读取的块大小，避免大文件一次性读入内存（doc/ 下有 24MB 的 PDF）
CHUNK_SIZE = 1024 * 1024


def calc_file_hash(file_path: str) -> str:
    """
    计算文件的 SHA-256（分块读取，避免大文件占满内存）
    :param file_path: 文件绝对路径
    :return: 64位十六进制哈希字符串
    """
    function_name = sys._getframe().f_code.co_name
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
    file_hash = sha256.hexdigest()
    logger.info(f"[{function_name}] 文件哈希计算完成：{os.path.basename(file_path)} -> {file_hash[:16]}...")
    return file_hash


if __name__ == '__main__':
    """本地测试：对指定文件计算哈希"""
    import sys as _sys
    if len(_sys.argv) > 1:
        logger.info(calc_file_hash(_sys.argv[1]))
    else:
        logger.info("用法：python -m app.utils.file_hash_utils <文件路径>")
