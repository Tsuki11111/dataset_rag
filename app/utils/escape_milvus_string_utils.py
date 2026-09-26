# ===================== 核心辅助函数 =====================
def escape_milvus_string(value: str) -> str:
    """
    Milvus过滤表达式专用字符串安全转义函数
    核心作用：
        避免因原始字符串含特殊字符，导致Milvus解析filter_expr时报错，保证CRUD操作正常执行
    转义规则：
        1. 反斜杠（\）→ 双反斜杠（\\）：Milvus表达式转义规则
        2. 双引号（"）→ 转义双引号（\"）：避免截断字符串表达式
        3. 换行/回车/制表符 → 空格：防止表达式换行导致解析失败
    参数：
        value: 需要转义的原始字符串（如产品名称、文件标题）
    返回：
        str: 转义后的安全字符串，可直接用于Milvus的filter_expr
    """
    if value is None:
        return ""
    # 确保输入为字符串类型，避免非字符串值报错
    s = str(value)
    # 按Milvus规则转义特殊字符
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    # 替换换行/回车/制表符为空格，保证表达式单行有效
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s


def build_item_name_filter(item_names) -> str:
    """
    构造「按产品名限定检索范围」的 Milvus 过滤表达式

    形如：item_name in ["产品A", "产品B"]

    每个产品名都经过转义 —— 产品名由大模型生成，可能含引号等字符，
    不转义会让整个过滤表达式解析失败（检索直接报错）。
    :param item_names: 产品名列表
    :return: 过滤表达式；无有效产品名时返回 ""，调用方据此不做过滤
    """
    if not item_names:
        return ""
    # 先剔除空值再转义，避免拼出 item_name in ["", "A"] 这种表达式
    names = [str(v).strip() for v in item_names if v and str(v).strip()]
    if not names:
        return ""
    quoted = ", ".join(f'"{escape_milvus_string(v)}"' for v in names)
    return f"item_name in [{quoted}]"