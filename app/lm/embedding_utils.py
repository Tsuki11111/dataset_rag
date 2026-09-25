from openai import OpenAI
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 文本上限参考：https://help.aliyun.com/zh/model-studio/text-embedding-api-reference
MAX_INPUT_LENGTH = 2048

# 客户端单例对象，避免重复初始化
_client = None


def get_client():
    """
    获取DashScope embedding客户端单例对象，自动加载环境变量配置
    :return: 初始化完成的OpenAI客户端实例
    """
    global _client
    if _client is not None:
        return _client

    if not embedding_config.api_key:
        logger.error("未配置DashScope API Key，请设置EMBEDDING_API_KEY或OPENAI_API_KEY")
        raise ValueError("缺少DashScope API Key")

    logger.info(
        "开始初始化DashScope embedding客户端",
        extra={
            "base_url": embedding_config.base_url,
            "model": embedding_config.model,
            "dimension": embedding_config.dimension,
        },
    )
    try:
        _client = OpenAI(api_key=embedding_config.api_key, base_url=embedding_config.base_url)
        logger.success("DashScope embedding客户端初始化成功")
        return _client
    except Exception as e:
        logger.error(f"DashScope embedding客户端初始化失败：{str(e)}", exc_info=True)
        raise

def _truncate(text):
    """
    将超长文本截断到MAX_INPUT_LENGTH字符，避免超出DashScope单条文本长度上限
    """
    if len(text) > MAX_INPUT_LENGTH:
        truncated = text[:MAX_INPUT_LENGTH]
        logger.warning(f"文本超长（{len(text)}字），已截断到{MAX_INPUT_LENGTH}字")
        return truncated
    return text


def _batch_encode(client, texts):
    """
    分批调用DashScope embedding接口，返回嵌套列表，与输入文本一一对应
    """
    embeddings = []
    batch_size = embedding_config.batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.debug(f"正在编码第{i // batch_size + 1}批，共{len(batch)}条")
        response = client.embeddings.create(model=embedding_config.model, input=batch)
        # 按索引排序，保证顺序与输入一致（API可能乱序返回）
        sorted_emb = sorted(response.data, key=lambda d: d.index)
        embeddings.extend([item.embedding for item in sorted_emb])
    return embeddings


def generate_embeddings(texts):
    """
    为文本列表生成稠密向量嵌入（DashScope text-embedding-v2，OpenAI兼容端点）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense，对应嵌套列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成稠密向量嵌入")
    try:
        client = get_client()
        texts = [_truncate(t) for t in texts]
        dense = _batch_encode(client, texts)

        result = {"dense": dense}  # 嵌套列表，与输入文本一一对应
        logger.success(f"{len(texts)}条文本向量生成完成，维度={embedding_config.dimension}")
        return result

    except Exception:
        logger.error("文本向量生成失败", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


"""
切换说明（由BGE-M3本地模型改为DashScope text-embedding-v2）：
1. 调用方式：改为DashScope OpenAI兼容端点（https://dashscope.aliyuncs.com/compatible-mode/v1），复用已配置的OPENAI_API_KEY/BASE_URL；
2. 返回结构：DashScope仅返回稠密向量，返回{"dense": [...]}，不再包含sparse稀疏向量，调用方取值逻辑相应调整；
3. 维度一致性：text-embedding-v2固定1536维，与.env中EMBEDDING_DIM一致，Milvus集合schema无需重建（如需切换其他模型须同步修改dimension并重建集合）；
4. 归一化：text-embedding-v2的余弦相似度检索无需显式归一化，MILVUS_METRIC_TYPE=COSINE可直接使用；
5. 文本上限：单条文本最长2048字符，超长自动截断；
6. 批处理：批量调用embedding接口，减少请求次数，默认每批16条（EMBEDDING_BATCH_SIZE可调）；
7. 单例模式：客户端仅初始化一次，复用连接提升批量处理效率；
8. 分级日志：从客户端初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
9. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""
