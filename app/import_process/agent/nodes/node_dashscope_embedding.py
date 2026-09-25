import sys
from typing import Dict, List

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_dashscope_embedding"


def step_1_validate_input(state: ImportGraphState) -> List[Dict]:
    """
    步骤 1: 校验输入数据
    从状态中提取待向量化的chunks，校验类型与非空性
    :param state: 流程状态字典
    :return: 校验通过的切片列表
    """
    function_name = sys._getframe().f_code.co_name
    texts_to_embed = state.get("chunks")
    if not isinstance(texts_to_embed, list) or not texts_to_embed:
        logger.error(f"[{NODE_NAME}] [{function_name}] 向量化输入校验失败：chunks字段为空或非有效列表")
        raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")

    logger.info(f"[{NODE_NAME}] [{function_name}] 向量化输入校验通过，待处理文本切片数量：{len(texts_to_embed)}")
    return texts_to_embed


def step_2_build_embed_texts(texts_to_embed: List[Dict]) -> List[str]:
    """
    步骤 2: 为每个切片构建送入嵌入模型的文本
    把产品名前置拼接：基于BERT架构的嵌入模型对前128个token注意力最集中，核心词前置可强化特征
    :param texts_to_embed: 切片列表
    :return: 与切片一一对应的待嵌入文本列表
    """
    function_name = sys._getframe().f_code.co_name
    input_texts = []
    empty_item_name_count = 0

    for doc in texts_to_embed:
        item_name = (doc.get("item_name") or "").strip()
        content = doc.get("content") or ""
        if not item_name:
            empty_item_name_count += 1
        # 有产品名则前置拼接，无则直接用内容
        input_texts.append(f"产品：{item_name}，介绍：{content}" if item_name else content)

    if empty_item_name_count:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 有{empty_item_name_count}个切片缺少item_name，已降级为仅嵌入正文")
    logger.info(f"[{NODE_NAME}] [{function_name}] 嵌入文本构建完成，共{len(input_texts)}条")
    return input_texts


def step_3_generate_embeddings(input_texts: List[str]) -> List[List[float]]:
    """
    步骤 3: 批量生成稠密向量
    说明：DashScope的text-embedding-v2只输出稠密向量，本项目不使用稀疏向量。
    embedding_utils内部已按EMBEDDING_BATCH_SIZE分批请求，此处无需再做外部分批。
    :param input_texts: 待嵌入文本列表
    :return: 与输入一一对应的稠密向量列表
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始生成稠密向量，共{len(input_texts)}条")
    try:
        vector_result = generate_embeddings(input_texts)
    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 向量生成失败：{str(e)}", exc_info=True)
        raise

    dense_list = (vector_result or {}).get("dense") or []
    if len(dense_list) != len(input_texts):
        logger.error(
            f"[{NODE_NAME}] [{function_name}] 向量数量与输入不匹配："
            f"输入{len(input_texts)}条，返回{len(dense_list)}条"
        )
        raise ValueError(f"向量生成结果数量不匹配：输入{len(input_texts)}条，返回{len(dense_list)}条")

    logger.success(f"[{NODE_NAME}] [{function_name}] 稠密向量生成完成，共{len(dense_list)}条，维度={len(dense_list[0])}")
    return dense_list


def step_4_attach_vectors(texts_to_embed: List[Dict], dense_list: List[List[float]]) -> List[Dict]:
    """
    步骤 4: 将向量回写到切片
    复制原切片后新增dense_vector字段，不修改上游源数据；
    保留原切片全部字段（content/title/parent_title/part/file_title/item_name），供下游入库节点使用
    :param texts_to_embed: 原始切片列表
    :param dense_list: 与切片一一对应的稠密向量
    :return: 带dense_vector字段的切片列表
    """
    function_name = sys._getframe().f_code.co_name
    output_data = []
    for doc, dense_vector in zip(texts_to_embed, dense_list):
        item = doc.copy() if isinstance(doc, dict) else {"content": str(doc)}
        item["dense_vector"] = dense_vector
        output_data.append(item)

    logger.info(f"[{NODE_NAME}] [{function_name}] 向量回写完成，共{len(output_data)}个切片已绑定dense_vector")
    return output_data


def node_dashscope_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_dashscope_embedding)
    为什么叫这个名字: 使用 DashScope 的 text-embedding-v2 模型将文本转换为向量 (Embedding)。
    整体流程：校验输入→构建嵌入文本→生成稠密向量→回写切片
    说明：DashScope text-embedding-v2 只输出稠密向量，本项目不使用稀疏向量
    :param state: 项目状态字典（ImportGraphState），需包含chunks/task_id
    :return: 更新后的状态字典，chunks中每个元素新增dense_vector字段
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{function_name}] 节点开始执行")
    add_running_task(state.get("task_id", ""), function_name)

    try:
        # 步骤1：校验输入数据
        texts_to_embed = step_1_validate_input(state)

        # 步骤2：构建送入嵌入模型的文本（产品名前置）
        input_texts = step_2_build_embed_texts(texts_to_embed)

        # 步骤3：批量生成稠密向量
        dense_list = step_3_generate_embeddings(input_texts)

        # 步骤4：将向量回写到切片
        state["chunks"] = step_4_attach_vectors(texts_to_embed, dense_list)

        logger.info(f"[{NODE_NAME}] [{function_name}] 节点执行完成，共{len(state['chunks'])}个切片已向量化")
    except Exception as e:
        logger.error(f"[{function_name}] 节点执行失败，错误信息：{str(e)}", exc_info=True)
        raise e
    finally:
        add_done_task(state.get("task_id", ""), function_name)

    return state


if __name__ == '__main__':
    """
    本地测试入口：依赖已切分并完成产品名识别的chunks，无需MinIO/PDF/Milvus
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

        # chunks.json 是文档切分节点的产物，不含item_name，这里模拟节点5的回填结果
        for chunk in test_chunks:
            chunk.setdefault("item_name", "Brother HAK 180 烫金机")

        test_state = {
            "task_id": "test_task_embedding_001",
            "chunks": test_chunks,
            "file_title": "hak180使用说明书",
        }
        result_state = node_dashscope_embedding(test_state)
        result_chunks = result_state.get("chunks", [])

        logger.info(f"[{NODE_NAME}] [__main__] 待处理切片数：{len(test_chunks)} | 实际处理切片数：{len(result_chunks)}")
        with_vector = sum(1 for c in result_chunks if c.get("dense_vector"))
        logger.info(f"[{NODE_NAME}] [__main__] 已绑定dense_vector的切片数：{with_vector}")
        first = result_chunks[0]
        logger.info(f"[{NODE_NAME}] [__main__] 首个切片字段：{list(first.keys())}")
        logger.info(f"[{NODE_NAME}] [__main__] 首个切片向量维度：{len(first.get('dense_vector') or [])}")
