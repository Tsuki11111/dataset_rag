import sys

from pymilvus import DataType, MilvusClient

from app.conf.embedding_config import embedding_config
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 切片文本字段的最大长度（Milvus VARCHAR 上限 65535）
TEXT_FIELD_MAX_LENGTH = 65535
# item_name 集合的主键长度：产品名称作为主键，512 足够且索引更省空间
ITEM_NAME_KEY_MAX_LENGTH = 512
# 建库与检索必须使用同一度量方式，COSINE 与 .env 的 MILVUS_METRIC_TYPE 保持一致
METRIC_TYPE = "COSINE"


def step1_validate_config() -> int:
    """
    1.校验建集合所需的配置
    :return: 向量维度
    """
    function_name = sys._getframe().f_code.co_name
    if not milvus_config.item_name_collection:
        logger.error(f"[{function_name}] 缺少ITEM_NAME_COLLECTION配置")
        raise Exception("缺少ITEM_NAME_COLLECTION配置")
    if not milvus_config.chunks_collection:
        logger.error(f"[{function_name}] 缺少CHUNKS_COLLECTION配置")
        raise Exception("缺少CHUNKS_COLLECTION配置")
    dimension = embedding_config.dimension
    if not dimension or dimension <= 0:
        logger.error(f"[{function_name}] 向量维度不合法: {dimension}")
        raise Exception(f"向量维度不合法: {dimension}")
    logger.info(f"[{function_name}] 配置校验通过，向量维度={dimension}")
    return dimension


def build_item_name_schema(client, dimension: int):
    """
    构建 kb_item_names 的 schema
    字段：item_name(主键) / file_title / dense_vector
    item_name 作为主键：产品名称天然唯一，重复导入时可用upsert直接覆盖，实现幂等。
    注意：upsert 是「标记删除 + 插入」，get_collection_stats 的 row_count 会包含已删除记录，
    判断实际记录数请用 query，不要用 row_count。
    :param client: MilvusClient实例
    :param dimension: 向量维度
    :return: CollectionSchema
    """
    function_name = sys._getframe().f_code.co_name
    # enable_dynamic_field=True：后续想加字段不必重建集合
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # 主键：产品名称本身，不再额外引入自增id
    schema.add_field(
        field_name="item_name",
        datatype=DataType.VARCHAR,
        max_length=ITEM_NAME_KEY_MAX_LENGTH,
        is_primary=True,
    )
    # 产品名称所属的源文件标题，便于反查产品来自哪篇文档
    schema.add_field(
        field_name="file_title",
        datatype=DataType.VARCHAR,
        max_length=TEXT_FIELD_MAX_LENGTH,
    )
    # item_name生成的稠密向量（DashScope text-embedding-v2，仅稠密）
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )
    logger.info(f"[{function_name}] schema构建完成，主键=item_name，向量维度={dimension}")
    return schema


def build_chunks_schema(client, dimension: int):
    """
    构建 kb_chunks 的 schema
    字段：chunk_id(自增主键) / content / title / parent_title / part / file_title / item_name / dense_vector
    与文档一致，但去掉稀疏向量（本项目只用DashScope稠密向量）
    chunk_id 用自增主键：同一个产品名会被切成多个切片，主键不能是业务字段
    :param client: MilvusClient实例
    :param dimension: 向量维度
    :return: CollectionSchema
    """
    function_name = sys._getframe().f_code.co_name
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 自增主键：入库后由Milvus生成，并回填到切片供下游使用
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    # 切片正文，可能较长（当前最大约2400字符），给足长度避免插入失败
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=TEXT_FIELD_MAX_LENGTH)
    # 切片标题（形如 "## 1.2 产品简介"）
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=TEXT_FIELD_MAX_LENGTH)
    # 父标题：合并短切片时的归并依据，也是切片所属章节的标识
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=TEXT_FIELD_MAX_LENGTH)
    # 分片编号：长段落被二次切分时的序号（1/2/...），对应切分节点的 part
    schema.add_field(field_name="part", datatype=DataType.INT8)
    # 源文件标题
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=TEXT_FIELD_MAX_LENGTH)
    # 产品名称：入库前按此字段做幂等清理
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=TEXT_FIELD_MAX_LENGTH)
    # 切片正文的稠密向量（DashScope text-embedding-v2，仅稠密）
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    logger.info(f"[{function_name}] schema构建完成，主键=chunk_id(自增)，向量维度={dimension}")
    return schema


def step3_build_index_params(client):
    """
    3.构建索引参数
    稠密向量用HNSW索引（检索快、召回高），度量方式与建库/检索保持一致
    无稀疏向量，因此不需要 SPARSE_INVERTED_INDEX
    :param client: MilvusClient实例
    :return: IndexParams
    """
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type=METRIC_TYPE,
        # M: 图中每个节点的最大连接数；efConstruction: 建索引时的搜索范围
        params={"M": 16, "efConstruction": 200},
    )
    logger.info(f"[{sys._getframe().f_code.co_name}] 索引参数构建完成，index_type=HNSW，metric_type={METRIC_TYPE}")
    return index_params


def step4_create_collection(client, collection_name, schema, index_params) -> bool:
    """
    4.创建集合并加载到内存
    已存在则跳过，避免重复建库覆盖数据
    :return: True表示本次新建，False表示已存在跳过
    """
    function_name = sys._getframe().f_code.co_name
    if client.has_collection(collection_name):
        logger.warning(f"[{function_name}] 集合[{collection_name}]已存在，跳过创建")
        return False
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    # 加载后才能被检索到
    client.load_collection(collection_name)
    logger.success(f"[{function_name}] 集合[{collection_name}]创建并加载完成")
    return True


def create_item_name_collection(client, dimension: int) -> None:
    """
    创建 kb_item_names 集合（产品主体名称）
    """
    collection_name = milvus_config.item_name_collection
    schema = build_item_name_schema(client, dimension)
    index_params = step3_build_index_params(client)
    step4_create_collection(client, collection_name, schema, index_params)


def create_chunks_collection(client, dimension: int) -> None:
    """
    创建 kb_chunks 集合（文档切片）
    """
    collection_name = milvus_config.chunks_collection
    schema = build_chunks_schema(client, dimension)
    index_params = step3_build_index_params(client)
    step4_create_collection(client, collection_name, schema, index_params)


def create_all_collections() -> None:
    """
    创建全部集合（kb_item_names + kb_chunks）
    流程：校验配置 -> 连接Milvus -> 逐个建表并加载
    说明：文档指纹（去重记录）存在 SQLite 里，不是 Milvus 集合
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{function_name}] 开始创建Milvus集合")

    try:
        # 1.校验配置
        dimension = step1_validate_config()
        client = MilvusClient(uri=milvus_config.milvus_url)
        # 2.建 kb_item_names
        create_item_name_collection(client, dimension)
        # 3.建 kb_chunks
        create_chunks_collection(client, dimension)
        logger.info(f"[{function_name}] 集合创建流程结束，当前集合列表: {client.list_collections()}")
    except Exception as e:
        logger.error(f"[{function_name}] 创建失败，错误信息为{e}", exc_info=True)
        raise e


if __name__ == '__main__':
    """
    本地测试入口：需要本地Milvus已启动
    """
    create_all_collections()
