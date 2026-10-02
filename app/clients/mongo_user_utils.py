"""
用户与访问密钥读写工具（基于 MongoDB）

用途：提供「你是谁」这一层——请求带上密钥，服务校验后知道调用方身份。

**为什么现在只做认证、不做数据隔离**：数据隔离要给 Milvus / Neo4j / MongoDB / MinIO
四个存储都加租户维度，属于停机迁移级别的改动。这里先把身份层打下来，
`tenant_id` 随用户一起发出来，等真要隔离时直接往各存储写即可，不必返工。

**密钥只存 sha256 哈希**：密钥是 32 字节随机串、熵足够高，不需要 bcrypt 那类抗暴力破解的
慢哈希；而每个请求都要校验一次，慢哈希会平白多出几十毫秒。

设计沿用 mongo_dedup_utils / mongo_history_utils 的模式：工具类 + 全局单例 + 懒加载。
集合名 users，key_hash 建唯一索引。
"""
import hashlib
import os
import secrets
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

from app.core.logger import logger

# 集合名
COLLECTION_NAME = "users"

# 角色：先只有两种，够用很久
ROLE_ADMIN = "admin"    # 可上传、可撤回
ROLE_VIEWER = "viewer"  # 只能查询
ROLES = (ROLE_ADMIN, ROLE_VIEWER)

# 密钥字节数：32 字节 → 64 位十六进制，足够抗猜测
KEY_BYTES = 32

_user_tool = None


def _hash_key(raw_key: str) -> str:
    """密钥哈希。只存哈希不存明文，库泄露时拿不到可用的密钥"""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class UserTool:
    """
    MongoDB 用户表读写工具类

    封装连接、集合获取与索引创建，为上层提供统一的用户操作入口。
    """

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.collection = self.db[COLLECTION_NAME]

            # key_hash 唯一：同一个密钥不会同时发给两个人
            self.collection.create_index([("key_hash", 1)], unique=True)
            # tenant_id 建普通索引（不唯一）：目前一人一租户，将来一个租户下可挂多人
            self.collection.create_index([("tenant_id", 1)])
        except Exception as e:
            logger.error(f"用户工具初始化失败：{e}", exc_info=True)
            raise


def get_user_tool() -> UserTool:
    """获取用户工具单例（懒加载）"""
    global _user_tool
    if _user_tool is None:
        _user_tool = UserTool()
    return _user_tool


def count_users() -> int:
    """当前用户数。服务启动时用它判断是否需要提示建号"""
    try:
        return get_user_tool().collection.count_documents({"revoked_at": None})
    except Exception as e:
        logger.error(f"统计用户数失败：{e}", exc_info=True)
        return 0


def create_user(name: str, role: str = ROLE_ADMIN, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """
    新建用户并生成一把新密钥

    **明文密钥只在返回值里出现这一次**，库里只有哈希，丢了只能重建。
    :param name: 人可读的称呼
    :param role: admin / viewer
    :param tenant_id: 留空则自动生成（目前一人一租户）
    :return: 用户信息 + {"raw_key": 明文密钥}
    """
    if role not in ROLES:
        raise ValueError(f"未知角色 {role!r}，可选：{ROLES}")

    raw_key = secrets.token_hex(KEY_BYTES)
    doc = {
        "tenant_id": tenant_id or f"t_{secrets.token_hex(4)}",
        "name": name,
        "role": role,
        "key_hash": _hash_key(raw_key),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "revoked_at": None,
        "last_used_at": None,
    }
    get_user_tool().collection.insert_one(doc)
    logger.info(f"已创建用户[{name}]，角色={role}，租户={doc['tenant_id']}")

    doc.pop("_id", None)
    doc.pop("key_hash", None)
    return {**doc, "raw_key": raw_key}


def verify_key(raw_key: str) -> Optional[Dict[str, Any]]:
    """
    校验访问密钥

    :param raw_key: 调用方提交的明文密钥
    :return: 有效则返回 {tenant_id, name, role}，无效或已撤销返回 None
    """
    if not raw_key:
        return None

    try:
        tool = get_user_tool()
        doc = tool.collection.find_one({"key_hash": _hash_key(raw_key), "revoked_at": None})
        if not doc:
            return None

        # 记录最近使用时间，便于排查「这把钥匙还有没有人在用」。
        # 用单独的 update 且吞掉异常：记录时间失败不该让鉴权失败
        try:
            tool.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"last_used_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}},
            )
        except Exception as e:
            logger.warning(f"更新密钥使用时间失败（不影响鉴权）：{e}")

        return {
            "tenant_id": doc.get("tenant_id"),
            "name": doc.get("name"),
            "role": doc.get("role"),
        }
    except Exception as e:
        logger.error(f"校验密钥失败：{e}", exc_info=True)
        return None


def list_users() -> List[Dict[str, Any]]:
    """列出所有用户（不含密钥哈希）"""
    try:
        rows = list(get_user_tool().collection.find({}, {"key_hash": 0}))
        for r in rows:
            r.pop("_id", None)
        return rows
    except Exception as e:
        logger.error(f"列出用户失败：{e}", exc_info=True)
        return []


def revoke_user(name: str) -> int:
    """
    按称呼撤销用户（软删除，保留记录便于审计）

    :return: 被撤销的条数
    """
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res = get_user_tool().collection.update_many(
            {"name": name, "revoked_at": None},
            {"$set": {"revoked_at": now}},
        )
        if res.modified_count:
            logger.warning(f"已撤销用户[{name}]，共 {res.modified_count} 条")
        return res.modified_count
    except Exception as e:
        logger.error(f"撤销用户[{name}]失败：{e}", exc_info=True)
        return 0


if __name__ == '__main__':
    """
    命令行建号 / 管理

    用法：
      .venv/Scripts/python.exe -m app.clients.mongo_user_utils add 张三 [admin|viewer]
      .venv/Scripts/python.exe -m app.clients.mongo_user_utils list
      .venv/Scripts/python.exe -m app.clients.mongo_user_utils revoke 张三
    """
    args = sys.argv[1:]
    action = args[0] if args else "list"

    if action == "add":
        if len(args) < 2:
            print("用法：add <称呼> [admin|viewer]")
            raise SystemExit(1)
        name = args[1]
        role = args[2] if len(args) > 2 else ROLE_ADMIN
        user = create_user(name, role)
        print("\n用户已创建，请立刻保存下面这把密钥——库里只有哈希，丢了只能重建：\n")
        print(f"  称呼   ：{user['name']}")
        print(f"  角色   ：{user['role']}")
        print(f"  租户   ：{user['tenant_id']}")
        print(f"  密钥   ：{user['raw_key']}\n")
    elif action == "revoke":
        if len(args) < 2:
            print("用法：revoke <称呼>")
            raise SystemExit(1)
        print(f"已撤销 {revoke_user(args[1])} 条")
    else:
        users = list_users()
        if not users:
            print("还没有任何用户。用 add 子命令创建第一个：")
            print("  python -m app.clients.mongo_user_utils add 张三")
        for u in users:
            state = "已撤销" if u.get("revoked_at") else "有效"
            print(
                f"  {u.get('name'):<10} {u.get('role'):<7} {u.get('tenant_id'):<12} {state:<5} "
                f"创建于 {u.get('created_at')}  最近使用 {u.get('last_used_at') or '—'}"
            )
