import sys
from typing import Dict, List, Tuple

from langchain.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient

from app.clients.milvus_utils import get_milvus_client, upsert_item_name
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_item_name_recognition"

# --- 配置参数 (Configuration) ---
# 大模型识别产品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500
# 大模型未识别出产品名时可能输出的占位词（非真实产品名），命中的一律走file_title兜底
UNRECOGNIZED_PLACEHOLDERS = {
    "空字符串", "空", "无", "未知", "未知产品", "无法识别", "未能识别",
    "none", "null", "n/a", "na", "unknown", "",
}


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """
    步骤 1: 接收并校验流程输入
    从状态中提取文件标题与切片列表，做多层空值兜底，避免后续流程因空值报错
    :param state: 流程状态字典
    :return: (文件标题, 切片列表)；切片非法时返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    # 多层兜底获取文件标题：优先file_title → 其次file_name → 空字符串
    file_title = state.get("file_title", "") or state.get("file_name", "")
    # 获取文本切片列表：空值时返回空列表，避免后续遍历报错
    chunks = state.get("chunks") or []

    # 二次兜底：file_title仍为空时，尝试从第一个有效切片中提取
    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning(f"[{NODE_NAME}] [{function_name}] state中无有效file_title，已从第一个切片中提取兜底标题")

    if not file_title:
        logger.warning(f"[{NODE_NAME}] [{function_name}] state中缺少file_title和file_name，后续大模型识别可能精度下降")

    # 数据类型校验：确保chunks为有效非空列表
    if not isinstance(chunks, list) or not chunks:
        logger.warning(f"[{NODE_NAME}] [{function_name}] state中chunks为空或非列表类型，无法进行产品名称识别")
        return file_title, []

    logger.info(f"[{NODE_NAME}] [{function_name}] 输入校验完成，获取到{len(chunks)}个有效文本切片")
    return file_title, chunks


def step_2_build_context(
    chunks: List[Dict],
    k: int = DEFAULT_ITEM_NAME_CHUNK_K,
    max_chars: int = CONTEXT_TOTAL_MAX_CHARS,
) -> str:
    """
    步骤 2: 构造大模型产品名称识别的标准化上下文
    限制切片数量与字符长度，过滤无效切片，带序号格式化以提升识别精度
    :param chunks: 文本切片列表
    :param k: 最大取片数
    :param max_chars: 上下文总字符数上限
    :return: 格式化后的上下文字符串，空切片时返回空字符串
    """
    function_name = sys._getframe().f_code.co_name
    if not chunks:
        return ""

    parts: List[str] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks[:k]):
        if not isinstance(chunk, dict):
            logger.debug(f"[{NODE_NAME}] [{function_name}] 第{idx + 1}个切片非字典类型，已过滤")
            continue

        chunk_title = (chunk.get("title") or "").strip()
        chunk_content = (chunk.get("content") or "").strip()

        if not (chunk_title or chunk_content):
            logger.debug(f"[{NODE_NAME}] [{function_name}] 第{idx + 1}个切片为空白内容，已过滤")
            continue

        # 单切片内容截断：防止单个切片内容过长占满上下文
        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
            logger.debug(f"[{NODE_NAME}] [{function_name}] 第{idx + 1}个切片内容过长，已截断至{SINGLE_CHUNK_CONTENT_MAX_LEN}字符")

        piece = f"【切片{idx + 1}】\n标题：{chunk_title} \n内容：{chunk_content}"
        parts.append(piece)
        total_chars += len(piece)

        if total_chars > max_chars:
            logger.info(f"[{NODE_NAME}] [{function_name}] 上下文总字符数即将超限（{max_chars}），已停止拼接后续切片")
            break

    context = "\n\n".join(parts).strip()
    # 最终二次截断，确保绝对不超限
    final_context = context[:max_chars]
    logger.info(f"[{NODE_NAME}] [{function_name}] 上下文构建完成，最终长度{len(final_context)}字符")
    return final_context


def step_3_call_llm(file_title: str, context: str) -> str:
    """
    步骤 3: 调用大模型实现产品名称/型号精准识别
    上下文为空或大模型返回空/异常时，均用file_title兜底，保证流程不中断
    :param file_title: 文件标题，异常或空值时的兜底值
    :param context: 步骤2构建的结构化上下文
    :return: 清洗后的产品名称
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始执行步骤3：调用大模型识别产品名称")

    if not context:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 上下文为空，跳过大模型调用，直接使用文件标题作为产品名称")
        return file_title

    try:
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        system_prompt = load_prompt("product_recognition_system")
        logger.debug(f"[{NODE_NAME}] [{function_name}] 提示词构建完成，系统提示词长度{len(system_prompt)}，人类提示词长度{len(human_prompt)}")

        llm = get_llm_client(json_mode=False)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        resp = llm.invoke(messages)

        item_name = (getattr(resp, "content", "") or "").strip()
        # 只去掉换行/制表符，保留产品名内部的空格（如 "Brother HAK 180"）
        item_name = item_name.replace("\n", "").replace("\r", "").replace("\t", "")

        # 大模型可能把提示词里的「返回空字符串」当成字面量输出（实测返回过"空字符串"四个字），
        # 也可能自行输出"无"、"未知"等占位词。这些都不是真实产品名，统一判为未识别，
        # 否则会写进 kb_item_names 主键、污染幂等去重依据
        if item_name in UNRECOGNIZED_PLACEHOLDERS:
            logger.warning(
                f"[{NODE_NAME}] [{function_name}] 大模型返回占位词[{item_name}]而非真实产品名，"
                f"使用文件标题兜底"
            )
            return file_title

        if not item_name:
            logger.warning(f"[{NODE_NAME}] [{function_name}] 大模型返回空内容，使用文件标题作为产品名称兜底")
            return file_title

        logger.info(f"[{NODE_NAME}] [{function_name}] 大模型识别产品名称成功，结果为：{item_name}")
        return item_name

    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 大模型调用失败，原因：{str(e)}", exc_info=True)
        return file_title


def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], item_name: str) -> None:
    """
    步骤 4: 回填产品名称到流程状态和所有文本切片
    所有切片关联同一产品名称，保证后续向量入库、检索时的维度一致性
    :param state: 流程状态对象
    :param chunks: 校验后的文本切片列表
    :param item_name: 步骤3识别并清洗后的产品名称
    """
    function_name = sys._getframe().f_code.co_name
    state["item_name"] = item_name

    for chunk in chunks:
        if not isinstance(chunk, dict):
            logger.warning(f"[{NODE_NAME}] [{function_name}] 非法切片，跳过item_name回填：{chunk}")
            continue
        # 节点可能被重复执行，已有item_name的切片直接复用，避免覆盖
        if chunk.get("item_name"):
            continue
        chunk["item_name"] = item_name

    state["chunks"] = chunks
    logger.info(f"[{NODE_NAME}] [{function_name}] 产品名称回填完成，共为{len(chunks)}个切片添加item_name字段，值为：{item_name}")


def step_5_generate_vectors(item_name: str):
    """
    步骤 5: 为产品名称生成稠密向量
    本项目嵌入模型为DashScope text-embedding-v2，只输出稠密向量，不使用稀疏向量
    :param item_name: 步骤3识别的产品名称
    :return: 稠密向量列表；空值或异常时返回None
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始执行步骤5：为产品名称[{item_name}]生成向量")

    if not item_name:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 产品名称为空，跳过向量生成，返回空向量")
        return None

    try:
        vector_result = generate_embeddings([item_name])
        dense_list = vector_result.get("dense") if vector_result else None
        if dense_list:
            dense_vector = dense_list[0]
            logger.success(f"[{NODE_NAME}] [{function_name}] 产品名称向量生成成功，维度={len(dense_vector)}")
        else:
            logger.warning(f"[{NODE_NAME}] [{function_name}] 向量生成工具返回空结果，无法提取向量")
            dense_vector = None
    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 向量生成失败，原因：{str(e)}", exc_info=True)
        dense_vector = None

    return dense_vector


def step_6_save_to_milvus(
    state: ImportGraphState,
    file_title: str,
    item_name: str,
    dense_vector,
    client: MilvusClient | None = None,
) -> bool:
    """
    步骤 6: 将产品名称、文件标题、向量持久化到Milvus向量数据库
    幂等策略（两层）：
        1. item_name 是主键，同一产品名重复导入由 upsert 自动覆盖；
        2. 写入前按 file_title 清理同文档的旧记录——大模型每次输出的产品名可能抖动
           （实测仅差一个空格就会变成不同主键），不清理会让同一文档在库里分裂成多条。
    :param state: 流程状态对象，用于状态同步
    :param file_title: 处理后的文件标题
    :param item_name: 识别后的产品名称（主键去重依据）
    :param dense_vector: 步骤5生成的稠密向量
    :param client: 可复用的MilvusClient，不传则自行创建
    :return: True表示已写入，False表示跳过
    """
    function_name = sys._getframe().f_code.co_name
    collection_name = milvus_config.item_name_collection

    # 配置校验：任一关键项缺失都无法写入，跳过而不是抛异常，
    # 避免识别失败导致整个导入流程中断（后续chunk入库节点仍可继续）
    if not milvus_config.milvus_url or not collection_name:
        logger.warning(f"[{NODE_NAME}] [{function_name}] Milvus配置缺失（MILVUS_URL/ITEM_NAME_COLLECTION），跳过数据保存")
        return False
    if not item_name:
        logger.warning(f"[{NODE_NAME}] [{function_name}] item_name为空，跳过数据保存")
        return False
    if dense_vector is None:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 稠密向量为空，跳过数据保存")
        return False

    client = client or get_milvus_client()
    if not client:
        logger.error(f"[{NODE_NAME}] [{function_name}] 无法获取Milvus客户端（连接失败），跳过数据保存")
        return False

    try:
        if not client.has_collection(collection_name):
            logger.warning(f"[{NODE_NAME}] [{function_name}] 集合[{collection_name}]不存在，请先执行 create_collections.py 建库")
            return False

        # 幂等性处理1：删除同一文档的旧记录，避免产品名抖动导致同一文档堆积多条
        clean_file_title = (file_title or "").strip()
        if clean_file_title:
            safe_file_title = escape_milvus_string(clean_file_title)
            old_rows = client.query(
                collection_name=collection_name,
                filter=f'file_title=="{safe_file_title}"',
                output_fields=["item_name"],
            )
            # 只清理与本次名称不同的旧记录，同名记录交给下面的upsert覆盖
            stale_names = [
                row["item_name"] for row in old_rows
                if row.get("item_name") and row["item_name"] != item_name
            ]
            if stale_names:
                client.delete(
                    collection_name=collection_name,
                    ids=stale_names,
                )
                logger.info(f"[{NODE_NAME}] [{function_name}] 已清理文档[{clean_file_title}]的{len(stale_names)}条旧记录：{stale_names}")

        # 幂等性处理2：upsert：主键相同则覆盖，天然幂等
        """
        为什么用 upsert 而不是 insert
        item_name 是主键。upsert 的语义是「主键存在就覆盖，不存在就插入」，所以重复导入同一产品时不会产生重复行 —— 这是幂等的第一层。
        用 insert 反而会插入失败或产生重复（取决于主键冲突策略），所以文档里那种「自增 pk + 先 delete 再 insert」的写法我没沿用 —— 主键直接用 item_name 更简单，少一次往返。
        """
        upsert_item_name(client, collection_name, item_name, file_title, dense_vector)

        # 写入后回查确认，避免"以为写进去了"
        safe_item_name = escape_milvus_string(item_name)
        saved = client.query(
            collection_name=collection_name,
            filter=f'item_name=="{safe_item_name}"',
            output_fields=["item_name", "file_title"],
        )
        if saved:
            logger.success(f"[{NODE_NAME}] [{function_name}] 写入Milvus成功，已确认记录: {saved[0]}")
            state["item_name"] = item_name
            return True
        logger.warning(f"[{NODE_NAME}] [{function_name}] 写入后回查为空，item_name={item_name}")
        return False

    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 数据存入Milvus失败，原因：{str(e)}", exc_info=True)
        return False


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】产品主体名称识别（node_item_name_recognition）
    整体流程：提取输入→构建上下文→大模型识别→回填数据→生成向量→存入Milvus
    核心目的：利用大模型从文档切片中精准识别产品/主体名称，并把名称与向量存入数据库
    后续扩展点：支持多主体识别、增加产品属性提取、对接其他向量库等
    :param state: 项目状态字典（ImportGraphState），需包含chunks/file_title/task_id
    :return: 更新后的状态字典，新增item_name，且chunks中每个元素新增item_name字段
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【产品名称识别】{node_name}")
    add_running_task(state.get("task_id", ""), node_name)

    try:
        # 步骤1：提取并校验输入数据
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            logger.warning(f">>> 节点执行警告：{node_name}（无有效切片数据），跳过识别")
            return state

        # 步骤2：构建大模型识别上下文
        context = step_2_build_context(chunks)

        # 步骤3：调用大模型识别产品名称
        item_name = step_3_call_llm(file_title, context)

        # 步骤4：回填产品名称到状态和切片
        step_4_update_chunks(state, chunks, item_name)

        # 步骤5：生成向量（仅稠密向量）
        dense_vector = step_5_generate_vectors(item_name)
        state["item_name_embedding"] = [dense_vector] if dense_vector else []

        # 步骤6：存入Milvus向量数据库
        state["item_name_saved"] = step_6_save_to_milvus(
            state, file_title, item_name, dense_vector
        )

        logger.info(
            f">>> 核心节点执行完成：【产品名称识别】{node_name}，"
            f"识别结果：{item_name}，已存入Milvus：{state['item_name_saved']}"
        )
    except Exception as e:
        logger.error(f">>> 核心节点执行失败：【产品名称识别】{node_name}，错误信息：{str(e)}", exc_info=True)
        raise e
    finally:
        add_done_task(state.get("task_id", ""), node_name)

    return state


if __name__ == '__main__':
    """
    本地测试入口：依赖已切分好的chunks.json与本地Milvus，无需MinIO/PDF
    """
    import json
    import os

    from app.utils.path_util import PROJECT_ROOT

    # 切分节点的备份现在按文档隔离存放，这里指向 HAK180 那份
    test_chunks_path = os.path.join(PROJECT_ROOT, "output", "hak180使用说明书", "chunks.json")
    if not os.path.exists(test_chunks_path):
        logger.error(f"[{NODE_NAME}] [__main__] 本地测试 - 测试文件不存在：{test_chunks_path}")
    else:
        with open(test_chunks_path, 'r', encoding='utf-8') as f:
            test_chunks = json.load(f)
        test_state = {
            "task_id": "test_task_123456",
            "file_title": "hak180使用说明书",
            "chunks": test_chunks,
            "item_name": "",
            "item_name_embedding": [],
            "item_name_saved": False,
        }
        result_state = node_item_name_recognition(test_state)
        logger.info(f"[{NODE_NAME}] [__main__] 本地测试完成 - item_name={result_state['item_name']}")
        embedding = result_state['item_name_embedding']
        logger.info(f"[{NODE_NAME}] [__main__] 本地测试完成 - 向量条数={len(embedding)}，维度={len(embedding[0]) if embedding else 0}")
        logger.info(f"[{NODE_NAME}] [__main__] 本地测试完成 - 是否写入Milvus={result_state['item_name_saved']}")
