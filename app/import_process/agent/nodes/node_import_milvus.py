import sys
from typing import Any, Dict, List, Tuple

from pymilvus import MilvusClient

from app.clients.milvus_utils import get_milvus_client
from app.conf.embedding_config import embedding_config
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_import_milvus"
# 集合名称，与建表脚本 create_collections.py 保持一致
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection
# 切片的必需字段，缺任一个都无法入库（part 允许为0，单独判断）
REQUIRED_FIELDS = ["content", "title", "parent_title", "part", "file_title", "item_name", "dense_vector"]
# 入库批次大小：避免单次请求体过大
INSERT_BATCH_SIZE = 200


def step_1_check_input(state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """
    步骤 1: 输入数据有效性校验
    校验切片非空、必需字段齐全、向量维度与配置一致
    :param state: 流程状态字典
    :return: (校验通过的切片列表, 稠密向量维度)
    """
    function_name = sys._getframe().f_code.co_name
    chunks_json_data = state.get("chunks")

    if not chunks_json_data:
        logger.error(f"[{NODE_NAME}] [{function_name}] Milvus入库校验失败：state中chunks字段为空")
        raise ValueError("错误: chunks为空，无法执行Milvus入库")
    if not isinstance(chunks_json_data, list):
        logger.error(f"[{NODE_NAME}] [{function_name}] Milvus入库校验失败：chunks非列表类型")
        raise ValueError("错误: chunks数据格式不正确，必须为非空列表")

    # 逐条校验必需字段，指出具体的缺失位置便于排查
    for idx, chunk in enumerate(chunks_json_data):
        if not isinstance(chunk, dict):
            logger.error(f"[{NODE_NAME}] [{function_name}] Milvus入库校验失败：第{idx + 1}条切片非字典类型")
            raise ValueError(f"错误: 第{idx + 1}条切片格式不正确")
        missing = [
            f for f in REQUIRED_FIELDS
            if f not in chunk or (chunk[f] is None and f != "part") or (f == "part" and chunk.get("part") is None)
        ]
        if missing:
            logger.error(f"[{NODE_NAME}] [{function_name}] Milvus入库校验失败：第{idx + 1}条切片缺失字段{missing}")
            raise ValueError(f"错误: 第{idx + 1}条切片缺失字段{missing}，请检查上游向量化节点执行状态")

    # 向量维度一致性：以第一条为准，逐条比对，避免插入时维度冲突
    vector_dimension = len(chunks_json_data[0]["dense_vector"])
    inconsistent = [i + 1 for i, c in enumerate(chunks_json_data) if len(c["dense_vector"]) != vector_dimension]
    if inconsistent:
        logger.error(f"[{NODE_NAME}] [{function_name}] Milvus入库校验失败：向量维度不一致，问题切片序号{inconsistent[:10]}")
        raise ValueError(f"错误: 存在维度不一致的向量，问题切片序号{inconsistent[:10]}")

    # 与集合/嵌入模型配置比对：维度不符会导致插入直接失败，提前拦截给出明确提示
    if embedding_config.dimension and vector_dimension != embedding_config.dimension:
        logger.error(
            f"[{NODE_NAME}] [{function_name}] Milvus入库校验失败："
            f"向量维度{vector_dimension}与EMBEDDING_DIM配置{embedding_config.dimension}不一致"
        )
        raise ValueError(f"错误: 向量维度{vector_dimension}与配置{embedding_config.dimension}不一致，请检查嵌入模型配置")

    item_name = chunks_json_data[0].get("item_name", "未知产品名")
    logger.info(
        f"[{NODE_NAME}] [{function_name}] Milvus入库校验通过，待入库切片数：{len(chunks_json_data)} | "
        f"向量维度：{vector_dimension} | 产品名称：{item_name}"
    )
    return chunks_json_data, vector_dimension


def step_2_prepare_collection(vector_dimension: int) -> MilvusClient:
    """
    步骤 2: 获取Milvus客户端并检查集合是否就绪
    集合由 create_collections.py 创建，此处只检查不创建，避免流程中隐式建表
    :param vector_dimension: 稠密向量维度
    :return: MilvusClient实例
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始准备Milvus环境，目标集合：{CHUNKS_COLLECTION_NAME}")

    if not CHUNKS_COLLECTION_NAME:
        logger.error(f"[{NODE_NAME}] [{function_name}] 未配置CHUNKS_COLLECTION集合名称")
        raise ValueError("未配置CHUNKS_COLLECTION集合名称")

    client = get_milvus_client()
    if client is None:
        logger.error(f"[{NODE_NAME}] [{function_name}] Milvus客户端获取失败，连接可能异常")
        raise ValueError("Milvus 连接失败：get_milvus_client() 返回空")

    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        logger.error(
            f"[{NODE_NAME}] [{function_name}] 集合{CHUNKS_COLLECTION_NAME}不存在，"
            f"请先执行 create_collections.py 建库"
        )
        raise ValueError(f"集合{CHUNKS_COLLECTION_NAME}不存在，请先执行 create_collections.py")

    # 集合向量维度与本次数据比对，维度不符会在插入时报错，提前拦截
    described = client.describe_collection(CHUNKS_COLLECTION_NAME)
    for field in described.get("fields", []):
        if field.get("name") == "dense_vector":
            collection_dim = (field.get("params") or {}).get("dim")
            if collection_dim and int(collection_dim) != vector_dimension:
                logger.error(
                    f"[{NODE_NAME}] [{function_name}] 集合向量维度{collection_dim}与数据维度{vector_dimension}不一致"
                )
                raise ValueError(f"集合向量维度{collection_dim}与数据维度{vector_dimension}不一致")

    logger.info(f"[{NODE_NAME}] [{function_name}] 集合{CHUNKS_COLLECTION_NAME}已就绪")
    return client


def step_3_clean_old_data(client: MilvusClient, chunks_json_data: List[Dict[str, Any]]) -> int:
    """
    步骤 3: 幂等性处理 - 按file_title清理同一文档的历史切片
    以「文件」作为文档身份，而不是item_name：item_name是大模型的输出，会抖动
    （实测同一文档两次识别仅差一个空格就变成不同的item_name），用它当幂等键会导致
    同一文档在库里堆积多批数据；更要命的是不同文档可能描述同一产品，
    按item_name删会误删其他文档的切片。
    :param client: MilvusClient实例
    :param chunks_json_data: 待入库的切片列表
    :return: 本次清理的file_title个数
    """
    function_name = sys._getframe().f_code.co_name
    # 提取并去重file_title，顺带剔除空值
    file_titles = sorted({
        title
        for x in chunks_json_data or []
        if (title := str(x.get("file_title", "")).strip())
    })

    if not file_titles:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 切片中无有效file_title，跳过幂等性清理，可能导致重复数据")
        return 0
    if len(file_titles) > 1:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 本批数据含多个file_title，将逐个清理：{file_titles}")

    for file_title in file_titles:
        safe_file_title = escape_milvus_string(file_title)
        client.delete(
            collection_name=CHUNKS_COLLECTION_NAME,
            filter=f'file_title == "{safe_file_title}"',
        )
        logger.info(f"[{NODE_NAME}] [{function_name}] 已删除文档[{file_title}]的历史切片")

    # 强制落盘，确保删除立即生效，避免与后续插入交错
    client.flush(CHUNKS_COLLECTION_NAME)
    logger.info(f"[{NODE_NAME}] [{function_name}] Milvus幂等性清理完成，共清理{len(file_titles)}个文档")

    # 只告警不删除：多个item_name可能是识别结果不稳定，也可能是本文档本含多个产品。
    # 这些切片属于不同文档的数据，删除会造成数据丢失，因此交由人工判断
    item_names = sorted({
        name
        for x in chunks_json_data or []
        if (name := str(x.get("item_name", "")).strip())
    })
    if len(item_names) > 1:
        logger.warning(
            f"[{NODE_NAME}] [{function_name}] 本批数据含{len(item_names)}个不同item_name（可能是识别结果抖动）：{item_names}"
        )
    return len(file_titles)


def step_4_insert_data(client: MilvusClient, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    步骤 4: 批量插入切片并回填chunk_id
    chunk_id由Milvus自增生成，插入后回填到切片供下游使用（按文档约定回填为str）
    :param client: MilvusClient实例
    :param chunks_json_data: 待入库的切片列表
    :return: 回填了chunk_id的切片列表
    """
    function_name = sys._getframe().f_code.co_name
    # 只保留集合里定义的字段：切片可能带有临时字段，直接提交会插入失败
    allowed_fields = {"content", "title", "parent_title", "part", "file_title", "item_name", "dense_vector"}
    data_to_insert = [
        {k: v for k, v in chunk.items() if k in allowed_fields}
        for chunk in chunks_json_data
    ]

    all_ids: List[Any] = []
    total = len(data_to_insert)
    logger.info(f"[{NODE_NAME}] [{function_name}] 准备{total}条切片数据，开始批量插入")

    for i in range(0, total, INSERT_BATCH_SIZE):
        batch = data_to_insert[i:i + INSERT_BATCH_SIZE]
        result = client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=batch)
        batch_ids = result.get("ids", []) if isinstance(result, dict) else []
        all_ids.extend(batch_ids)
        logger.info(f"[{NODE_NAME}] [{function_name}] 第{i + 1}-{i + len(batch)}条插入完成，本次生成{len(batch_ids)}个chunk_id")

    # 落盘并加载，确保数据立即可查
    client.flush(CHUNKS_COLLECTION_NAME)
    client.load_collection(CHUNKS_COLLECTION_NAME)

    if len(all_ids) != total:
        # 数量不一致说明部分批次异常，但数据已经插入。原先整批跳过回填，
        # 会导致已入库的切片全部没有chunk_id，下游按id补全内容时直接断链。
        # 改为按能对齐的部分回填，并明确告警缺失范围。
        paired = min(len(all_ids), len(chunks_json_data))
        for chunk, chunk_id in zip(chunks_json_data[:paired], all_ids):
            chunk["chunk_id"] = str(chunk_id)
        logger.error(
            f"[{NODE_NAME}] [{function_name}] chunk_id数量不匹配："
            f"生成{len(all_ids)}个，切片{total}条。已回填前{paired}条，"
            f"其余{total - paired}条切片未绑定chunk_id（下游按chunk_id补全将失败）"
        )
        return chunks_json_data

    # 按文档约定回填为字符串，便于JSON序列化与前端展示
    for chunk, chunk_id in zip(chunks_json_data, all_ids):
        chunk["chunk_id"] = str(chunk_id)
    logger.info(f"[{NODE_NAME}] [{function_name}] chunk_id回填完成，共{len(all_ids)}个切片已绑定chunk_id")
    return chunks_json_data


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    整体流程：校验输入→检查集合→按item_name清理旧数据→批量插入并回填chunk_id
    :param state: 项目状态字典（ImportGraphState），需包含chunks/task_id
    :return: 更新后的状态字典，chunks中每个元素新增chunk_id字段
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{function_name}] 节点开始执行")
    add_running_task(state.get("task_id", ""), function_name)

    try:
        # 步骤1：输入数据有效性校验
        chunks_json_data, vector_dimension = step_1_check_input(state)

        # 步骤2：Milvus客户端连接 + 集合检查
        client = step_2_prepare_collection(vector_dimension)

        # 步骤3：幂等性处理，按item_name清理旧数据
        step_3_clean_old_data(client, chunks_json_data)

        # 步骤4：批量插入并回填chunk_id
        state["chunks"] = step_4_insert_data(client, chunks_json_data)

        logger.info(f"[{function_name}] 节点执行完成，共{len(state['chunks'])}个切片已入库集合{CHUNKS_COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"[{function_name}] 节点执行失败，错误信息：{str(e)}", exc_info=True)
        raise e
    finally:
        add_done_task(state.get("task_id", ""), function_name)

    return state


if __name__ == '__main__':
    """
    本地测试入口：串起切分→产品识别→向量化→入库，验证完整链路
    依赖：本地Milvus已启动、已执行create_collections.py建库
    """
    import json
    import os

    from app.utils.path_util import PROJECT_ROOT

    # 切分节点的备份现在按文档隔离存放，这里指向 HAK180 那份
    test_chunks_path = os.path.join(PROJECT_ROOT, "output", "hak180使用说明书", "chunks.json")
    if not os.path.exists(test_chunks_path):
        logger.error(f"[{NODE_NAME}] [__main__] 测试文件不存在：{test_chunks_path}")
    else:
        with open(test_chunks_path, 'r', encoding='utf-8') as f:
            test_chunks = json.load(f)

        # chunks.json 不含item_name和向量，这里模拟节点5、节点6的产出
        for chunk in test_chunks:
            chunk.setdefault("item_name", "Brother HAK 180 烫金机")

        from app.import_process.agent.nodes.node_dashscope_embedding import node_dashscope_embedding
        test_state = {"task_id": "test_task_milvus_001", "chunks": test_chunks}
        test_state = node_dashscope_embedding(test_state)

        # 入库
        result_state = node_import_milvus(test_state)
        result_chunks = result_state.get("chunks", [])
        with_id = sum(1 for c in result_chunks if c.get("chunk_id"))
        logger.info(f"[{NODE_NAME}] [__main__] 入库切片数：{len(result_chunks)} | 已回填chunk_id数：{with_id}")
        logger.info(f"[{NODE_NAME}] [__main__] 首个切片chunk_id={result_chunks[0].get('chunk_id')}")

        # 回查集合实际记录数
        client = get_milvus_client()
        rows = client.query(
            collection_name=CHUNKS_COLLECTION_NAME,
            filter='item_name != ""',
            output_fields=["chunk_id", "title", "item_name"],
            limit=3,
        )
        logger.info(f"[{NODE_NAME}] [__main__] 集合中回查到的切片样例：{rows}")
