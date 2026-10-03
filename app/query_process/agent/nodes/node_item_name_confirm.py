"""
产品确认节点 (node_item_name_confirm)

节点作用：判断用户问的是**哪个产品**，以便定位到对应手册。

流程（7 步）：
1. 取历史会话
2. 保存当前问题
3. LLM 提取产品名 + 改写问题（处理代词消解）
4. 产品名向量化，在 kb_item_names 中检索标准产品名
5. 按相似度对齐（≥0.85 确认 / 0.6~0.85 候选 / <0.6 丢弃）
6. 三分支处理：
   A 有确认产品 → 写入 state，继续检索
   B 只有候选   → 生成反问句写入 state['answer']，主图条件边直接跳到答案输出
   C 无任何匹配 → 生成拒识句写入 state['answer']，同上
7. 持久化历史记录

注意：本项目嵌入模型为 DashScope（仅稠密向量），因此第 4 步用 dense_search，
而非教程中的稠密+稀疏混合检索。
"""
import json
import sys
from typing import Any, Dict, List

from langchain.messages import HumanMessage, SystemMessage

from app.clients.milvus_utils import dense_search, get_milvus_client
from app.clients.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.error_policy import ErrorKind, degrade, degrade_dependency
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_item_name_confirm"

# --- 配置参数 ---
# 参与上下文的历史消息条数
HISTORY_LIMIT = 10
# 相似度阈值：≥该值认为与库中标准产品名是同一个（教程代码用 0.85）
CONFIRM_SCORE_THRESHOLD = 0.85
# 候选阈值：≥该值但低于确认阈值时，作为候选让用户选择
CANDIDATE_SCORE_THRESHOLD = 0.6
# 反问时最多列出的候选数量
MAX_CANDIDATE_OPTIONS = 3
# 检索时对每个产品名取回的匹配数
SEARCH_LIMIT = 5

# 向用户确认的提示语（分支 B）
CLARIFY_TEMPLATE = "您是想问以下哪个产品：{options}？请明确一下型号。"
# 无匹配时的拒识语（分支 C）
NO_MATCH_ANSWER = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"


def step_1_get_history(session_id: str) -> List[Dict[str, Any]]:
    """
    步骤 1: 取该会话的最近若干条历史记录，用于代词消解与上下文理解
    :param session_id: 会话ID
    :return: 历史消息列表，失败返回空列表
    """
    function_name = sys._getframe().f_code.co_name
    try:
        history = get_recent_messages(session_id, limit=HISTORY_LIMIT)
    except Exception as e:
        # 历史读取失败退化为无上下文，不中断流程；编程错误由 degrade 上抛
        history = degrade(NODE_NAME, "读取会话历史", [], e)
    logger.info(f"[{NODE_NAME}] [{function_name}] 取到{len(history)}条历史消息")
    return history


def step_2_save_user_message(session_id: str, original_query: str) -> str:
    """
    步骤 2: 先保存用户当前问题，拿到消息ID供后续步骤7补充改写结果
    :return: 消息ID；保存失败返回空字符串
    """
    function_name = sys._getframe().f_code.co_name
    try:
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"[{NODE_NAME}] [{function_name}] 用户消息已保存，ID={message_id}")
        return message_id
    except Exception as e:
        return degrade(NODE_NAME, "保存用户消息", "", e)


def step_3_extract_info(query: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    步骤 3: 用大模型提取产品名并改写问题

    两个产出：
    - item_names：用户问的产品名（可能多个），用于后续向量对齐
    - rewritten_query：把「这个怎么装」这类指代不明的口语问题，改写成
      「XXX 怎么安装」这样的独立完整问题，提升后续检索召回率

    :return: {"item_names": [...], "rewritten_query": "..."}；失败时产品名为空、改写回退为原问题
    """
    function_name = sys._getframe().f_code.co_name
    fallback = {"item_names": [], "rewritten_query": query}

    # 拼历史为「角色: 内容」的文本，供 LLM 做指代消解
    history_text = "".join(f"{m.get('role', '')}: {m.get('text', '')}\n" for m in history)

    try:
        # json_mode=True：让模型直接返回可解析的 JSON，避免额外剥壳
        client = get_llm_client(json_mode=True)
        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
        messages = [
            SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
            HumanMessage(content=prompt),
        ]
        response = client.invoke(messages)
        content = (response.content or "").strip()

        # 兜底：部分模型仍会用 ```json 包裹
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(content)
        # 字段兜底，避免模型漏字段导致下游 KeyError
        if not isinstance(result.get("item_names"), list):
            result["item_names"] = []
        if not result.get("rewritten_query"):
            result["rewritten_query"] = query

        logger.info(
            f"[{NODE_NAME}] [{function_name}] LLM提取完成："
            f"item_names={result['item_names']}，rewritten_query={result['rewritten_query']!r}"
        )
        return result

    except Exception as e:
        # LLM 失败或 JSON 解析失败都退化为「无产品名 + 原问题」，保证流程不中断
        return degrade(NODE_NAME, "LLM 提取产品名", fallback, e)


def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict[str, Any]]:
    """
    步骤 4: 把提取出的产品名向量化，在 kb_item_names 中检索库里的标准产品名

    批量生成向量以减少 API 调用；逐个检索以保证结果与产品名一一对应。
    本项目仅用稠密向量，故走 dense_search（教程的混合检索已在本项目移除）。

    :param item_names: step3 提取的产品名列表
    :return: [{"extracted_name": 提取名, "matches": [{"item_name": 标准名, "score": 相似度}]}]
    """
    function_name = sys._getframe().f_code.co_name
    results: List[Dict[str, Any]] = []

    client = get_milvus_client()
    if client is None:
        # 这条尤其要紧：产品名对齐挂掉会让所有提问都落到"拒识"，用户只会觉得"什么都查不到"
        return degrade_dependency(NODE_NAME, "产品名对齐", results, "Milvus 不可用")

    collection_name = milvus_config.item_name_collection
    if not collection_name:
        return degrade_dependency(NODE_NAME, "产品名对齐", results,
                                  "未配置 ITEM_NAME_COLLECTION", ErrorKind.BLOCKED)

    try:
        embeddings = generate_embeddings(item_names)
        dense_vectors = embeddings.get("dense") or []
    except Exception as e:
        return degrade(NODE_NAME, "产品名向量化", results, e)

    for idx, name in enumerate(item_names):
        try:
            if idx >= len(dense_vectors):
                logger.warning(f"[{NODE_NAME}] [{function_name}] 产品名[{name}]缺少对应向量，跳过")
                continue

            hits = dense_search(
                client, collection_name, dense_vectors[idx],
                limit=SEARCH_LIMIT, output_fields=["item_name"],
                search_params={"ef": 64},
            )
            matches = []
            if hits and hits[0]:
                for hit in hits[0]:
                    matches.append({
                        "item_name": (hit.get("entity") or {}).get("item_name"),
                        "score": hit.get("distance", 0.0),
                    })
            results.append({"extracted_name": name, "matches": matches})
            logger.info(f"[{NODE_NAME}] [{function_name}] [{name}] 检索到{len(matches)}个匹配")

        except Exception as e:
            # 单个产品名失败不影响其余产品名
            degrade(NODE_NAME, f"检索产品名[{name}]", None, e)

    return results


def step_5_align_item_names(query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    步骤 5: 按相似度把提取名对齐到库中的标准产品名

    规则（优先级 a > b > c > d）：
      a 只有一个匹配 ≥0.85        → 确认为该产品
      b 多个匹配 ≥0.85           → 优先取与提取名完全相同的，否则取最高分
      c 无 ≥0.85 但有 ≥0.6       → 取分数最高的前几个作为候选，交由用户选择
      d 无 ≥0.6                  → 确认与候选均为空（视为未找到）

    :return: {
        "confirmed_item_names": [...],   # 已确认的标准产品名
        "options": [...],                # 候选标准产品名
        "corrections": {提取名: 标准名},  # 发生的纠正，供 step6 同步修正改写问题
    }
    """
    function_name = sys._getframe().f_code.co_name
    confirmed: List[str] = []
    options: List[str] = []
    corrections: Dict[str, str] = {}

    for res in query_results:
        extracted_name = (res.get("extracted_name") or "").strip()
        matches = res.get("matches") or []
        if not matches:
            continue

        # 按分数降序，高分优先
        matches.sort(key=lambda x: x.get("score") or 0, reverse=True)
        high = [m for m in matches if (m.get("score") or 0) >= CONFIRM_SCORE_THRESHOLD]
        mid = [m for m in matches if (m.get("score") or 0) >= CANDIDATE_SCORE_THRESHOLD]

        # 规则 a：唯一高置信度
        if len(high) == 1:
            picked = high[0]["item_name"]
            confirmed.append(picked)
            if extracted_name and extracted_name != picked:
                corrections[extracted_name] = picked
            continue

        # 规则 b：多个高置信度，优先与提取名完全一致的那个
        if len(high) > 1:
            picked = next((m["item_name"] for m in high if m["item_name"] == extracted_name), None)
            if not picked:
                picked = high[0]["item_name"]
            confirmed.append(picked)
            if extracted_name and extracted_name != picked:
                corrections[extracted_name] = picked
            continue

        # 规则 c：无高置信度，取中置信度前几个作为候选
        if mid:
            options.extend(m["item_name"] for m in mid[:MAX_CANDIDATE_OPTIONS])
        # 规则 d：什么都不做

    # 去重但保持顺序（list(set()) 会打乱顺序，候选展示需要稳定）
    def _dedup(seq):
        seen, out = set(), []
        for x in seq:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    result = {
        "confirmed_item_names": _dedup(confirmed),
        "options": _dedup(options),
        "corrections": corrections,
    }
    logger.info(
        f"[{NODE_NAME}] [{function_name}] 对齐结果：确认={result['confirmed_item_names']}，"
        f"候选={result['options']}，纠正={corrections}"
    )
    return result


def step_6_check_confirmation(
    align_result: Dict[str, Any],
    session_id: str,
    history: List[Dict[str, Any]],
    rewritten_query: str,
) -> Dict[str, Any]:
    """
    步骤 6: 按对齐结果决定流程走向

    分支 A 有确认产品：回填 state，并给历史中缺产品名的消息补上（上下文一致性），继续检索
    分支 B 只有候选：生成反问句 → state['answer']，主图条件边会直接跳到答案输出
    分支 C 无匹配：生成拒识句 → state['answer']，同上

    另外修正一处不一致：step3 的 rewritten_query 是基于 LLM 当时提取的名字写的，
    若 step5 把名字对齐成了标准名（如 "HAK" → "Brother HAK 180 烫金机"），
    就把改写问题里的旧名换成标准名，避免后续检索用错名字。

    :return: 供 LangGraph 合并进 state 的字段字典
    """
    function_name = sys._getframe().f_code.co_name
    confirmed = align_result.get("confirmed_item_names") or []
    options = align_result.get("options") or []
    corrections = align_result.get("corrections") or {}

    # 用对齐后的标准名替换改写问题里的旧名
    for old, new in corrections.items():
        if old in rewritten_query:
            rewritten_query = rewritten_query.replace(old, new)
    if corrections:
        logger.info(f"[{NODE_NAME}] [{function_name}] 改写问题已同步纠正：{rewritten_query!r}")

    # 分支 A：确认了产品，继续走检索
    if confirmed:
        # 给历史中还没关联产品名的消息补上，保持上下文一致
        ids_to_update = [str(m["_id"]) for m in history if m.get("_id") and not m.get("item_names")]
        if ids_to_update:
            try:
                update_message_item_names(ids_to_update, confirmed)
                logger.info(f"[{NODE_NAME}] [{function_name}] 已为{len(ids_to_update)}条历史消息补上产品名")
            except Exception as e:
                # 历史回填失败不影响本次检索
                degrade(NODE_NAME, "回填历史产品名", None, e)

        logger.info(f"[{NODE_NAME}] [{function_name}] 分支A：已确认产品 {confirmed}")
        return {
            "item_names": confirmed,
            "rewritten_query": rewritten_query,
            "answer": "",   # 清空，避免残留答案让条件边误判为「已有答案」
        }

    # 分支 B：有候选，反问用户
    if options:
        answer = CLARIFY_TEMPLATE.format(options="、".join(options[:MAX_CANDIDATE_OPTIONS]))
        logger.info(f"[{NODE_NAME}] [{function_name}] 分支B：需用户确认，候选={options}")
        return {"item_names": [], "rewritten_query": rewritten_query, "answer": answer}

    # 分支 C：完全没匹配上
    logger.info(f"[{NODE_NAME}] [{function_name}] 分支C：未找到相关产品")
    return {"item_names": [], "rewritten_query": rewritten_query, "answer": NO_MATCH_ANSWER}


def step_7_write_history(
    answer: str,
    session_id: str,
    original_query: str,
    rewritten_query: str,
    item_names: List[str],
    message_id: str,
) -> None:
    """
    步骤 7: 持久化本轮交互

    1. 若产生了答案（分支 B/C 的反问或拒识），写一条助手消息
    2. 更新用户那条消息，补上改写后的问题与识别出的产品名
    """
    function_name = sys._getframe().f_code.co_name

    if answer:
        try:
            save_chat_message(session_id, "assistant", answer)
            logger.info(f"[{NODE_NAME}] [{function_name}] 助手消息已存档")
        except Exception as e:
            degrade(NODE_NAME, "助手消息存档", None, e)

    if message_id:
        try:
            save_chat_message(
                session_id, "user", original_query,
                rewritten_query, item_names, message_id=message_id,
            )
            logger.info(f"[{NODE_NAME}] [{function_name}] 用户消息已更新（改写结果+产品名）")
        except Exception as e:
            degrade(NODE_NAME, "用户消息更新", None, e)


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    节点入口：串联 7 个步骤

    :param state: 需包含 session_id / original_query / is_stream
    :return: 更新的字段（item_names / rewritten_query / answer / history）
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{NODE_NAME}] [{function_name}] 开始处理")
    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, function_name, is_stream)

    # 1. 取历史会话
    history = step_1_get_history(session_id)

    # 2. 先保存用户当前问题，拿到消息ID
    message_id = step_2_save_user_message(session_id, original_query)

    # 3. LLM 提取产品名 + 改写问题
    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names") or []
    rewritten_query = extract_res.get("rewritten_query") or original_query

    # 4 & 5. 有产品名才做向量检索与对齐
    align_result: Dict[str, Any] = {}
    if item_names:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info(f"[{NODE_NAME}] [{function_name}] 未提取到产品名，跳过向量对齐")

    # 6. 按对齐结果决定分支
    updates = step_6_check_confirmation(align_result, session_id, history, rewritten_query)

    # 7. 持久化
    step_7_write_history(
        answer=updates.get("answer", ""),
        session_id=session_id,
        original_query=original_query,
        rewritten_query=updates.get("rewritten_query", rewritten_query),
        item_names=updates.get("item_names", []),
        message_id=message_id,
    )

    # history 一并存入 state，供下游节点（如 node_answer_output）使用
    updates["history"] = history

    add_done_task(session_id, function_name, is_stream)
    logger.info(
        f"[{NODE_NAME}] [{function_name}] 处理结束："
        f"item_names={updates.get('item_names')}，是否直接出答案={bool(updates.get('answer'))}"
    )
    return updates


if __name__ == '__main__':
    """
    本地测试：验证三个分支

    前置：Milvus 与 MongoDB 已启动
    """
    import time

    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task

    # 构造一个库里不存在的会话，确保历史为空，测试结果可预期
    test_session = f"confirm_test_{int(time.time())}"

    cases = [
        ("分支A 精确命中", "Brother HAK 180 烫金机怎么用？"),
        ("分支A 部分命中", "万用表怎么测量电压？"),
        ("分支B 模糊候选", "HAK180 怎么安装烫金膜盒？"),
        ("分支C 查无此人", "小米15 的电池怎么换？"),
    ]

    for label, query in cases:
        logger.info("=" * 70)
        logger.info(f"[测试] {label}：{query}")
        st = create_query_default_state(
            session_id=test_session + "_" + label[:4],
            original_query=query,
            is_stream=False,
        )
        try:
            result = node_item_name_confirm(st)
            logger.info(f"[测试] 产品名: {result.get('item_names')}")
            logger.info(f"[测试] 改写后: {result.get('rewritten_query')!r}")
            if result.get("answer"):
                logger.info(f"[测试] 直接答复: {result['answer']}")
        except Exception as e:
            logger.error(f"[测试] 执行失败：{e}", exc_info=True)
        finally:
            clear_task(st["session_id"])

    logger.info("=" * 70)
    logger.info("[测试] 全部用例执行完毕")
