from langgraph.constants import END
from langgraph.graph import StateGraph

from app.core.logger import logger
from app.core.usage_tracker import add_tracked_node, usage_context
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

# 初始化状态图
builder = StateGraph(QueryGraphState)

# 注册所有节点
# 用 add_tracked_node 而非 add_node：多包一层归因，让节点内所有模型调用都记到该节点名下
# （账本里的 node 字段），节点内部代码一行都不用改
add_tracked_node(builder, "node_item_name_confirm", node_item_name_confirm)   # 确认产品名
builder.add_node("node_multi_search", lambda x: x)                   # 虚拟节点：多路搜索分叉点
add_tracked_node(builder, "node_search_embedding", node_search_embedding)     # 向量检索
add_tracked_node(builder, "node_search_embedding_hyde", node_search_embedding_hyde)  # HyDE 检索
add_tracked_node(builder, "node_query_kg", node_query_kg)                     # 图谱检索
add_tracked_node(builder, "node_web_search_mcp", node_web_search_mcp)         # 联网检索
builder.add_node("node_join", lambda x: {})                          # 虚拟节点：多路搜索合并点
add_tracked_node(builder, "node_rrf", node_rrf)                               # RRF 融合排序
add_tracked_node(builder, "node_rerank", node_rerank)                         # 重排序
add_tracked_node(builder, "node_answer_output", node_answer_output)           # 生成答案

# 虚拟节点的作用：作为流程的「分叉 / 合并中转站」，解决多分支流程的组织问题，本身无业务逻辑

# 设置入口节点
builder.set_entry_point("node_item_name_confirm")


def route_after_item_name_confirm(state: QueryGraphState) -> str:
    """
    产品名确认后的路由

    若 node_item_name_confirm 已产出 answer，说明它无法唯一确定产品：
    - 多选一（反问用户）：用户问得太模糊，库里匹配到多个型号且置信度都不足，
      节点会生成反问句写入 answer，让用户明确型号
    - 查无此人（拒绝回答）：用户问的产品库里没有，或匹配评分过低，
      节点会生成拒绝句写入 answer

    这两种情况都不需要再检索文档，直接输出即可。
    """
    if state.get("answer"):
        logger.info("[route_after_item_name_confirm] 已有answer，跳过检索直接输出")
        return "node_answer_output"
    return "node_multi_search"


# 1. 产品名确认 ->（条件分叉）多路搜索 / 直接输出
builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_name_confirm,
    path_map={
        "node_multi_search": "node_multi_search",
        "node_answer_output": "node_answer_output",
    },
)

# 2. 从分叉点并发执行四路检索
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")
builder.add_edge("node_multi_search", "node_query_kg")

# 3. 四路检索 -> 结果合并点
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")
builder.add_edge("node_query_kg", "node_join")

# 4. 合并 -> 融合排序 -> 重排 -> 生成 -> 结束
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

# 编译生成可执行的 Runnable 应用
query_app = builder.compile()


if __name__ == '__main__':
    """
    端到端流程测试：验证检索图骨架能否走通

    当前节点均为骨架实现（仅 sleep + 返回空结果），本测试关注的是**流程结构**：
    1. 图能编译
    2. 节点按预期顺序执行（分叉/合并生效）
    3. 条件路由正确（有 answer 时跳过检索直接输出）

    说明：四路检索是并发的，LangGraph 不保证它们的执行先后，因此断言采用
    「集合相等」而非「顺序相等」，只校验该跑的节点都跑了。
    """
    import time

    from app.query_process.agent.state import create_query_default_state
    from app.utils.task_utils import clear_task, get_done_task_list, get_task_status

    logger.info("=" * 70)
    logger.info("[检索图测试] 开始端到端流程测试")
    logger.info("=" * 70)

    # ---------- 场景1：正常检索流程（应走完四路检索 → RRF → 重排 → 生成）----------
    session_id = f"query_test_{int(time.time())}"
    init_state = create_query_default_state(
        session_id=session_id,
        original_query="烫金膜盒怎么安装？",
        is_stream=False,   # 关闭流式，避免打字机效果拖慢测试
    )

    logger.info(f"[检索图测试] 场景1：正常检索流程，session_id={session_id}")
    start = time.time()
    # 包一层记账/日志上下文：命令行跑图也给它一条 trace。
    # 不包的话节点名有归因（add_tracked_node 在图上）、trace 与租户却空着，
    # 日志串不成一条、账目也归不到「哪一次运行」——服务入口是包了的，这里要对齐。
    with usage_context(session_id=session_id) as acc:
        final_state = query_app.invoke(init_state)
    elapsed = time.time() - start
    logger.info(f"[检索图测试] 本次问答记账：{acc.text()}")

    done = get_done_task_list(session_id)
    logger.info(f"[检索图测试] 执行完成，耗时 {elapsed:.1f} 秒")
    logger.info(f"[检索图测试] 任务状态：{get_task_status(session_id)}")
    logger.info(f"[检索图测试] 已完成节点（{len(done)}个）：{done}")

    # 该场景下必须执行的节点（节点名会被 task_utils 转成中文）
    # 注意：node_multi_search / node_join 是虚拟节点（lambda），不记录任务进度，故不在此列
    expected_all = {
        "确认问题产品",
        "切片搜索",                    # node_search_embedding
        "切片搜索(假设性文档)",         # node_search_embedding_hyde
        "查询知识图谱",                 # node_query_kg
        "网络搜索",                     # node_web_search_mcp
        "倒排融合",                     # node_rrf
        "重排序",                       # node_rerank
        "生成答案",                     # node_answer_output
    }
    actual = set(done)
    problems = []

    missing = expected_all - actual
    if missing:
        problems.append(f"未执行的节点：{missing}")

    if not final_state.get("answer"):
        problems.append("最终 state 中没有 answer")
    if "embedding_chunks" not in final_state:
        problems.append("state 中缺少 embedding_chunks 字段")

    logger.info(f"[检索图测试] 最终 answer：{final_state.get('answer', '')[:60]}...")

    # ---------- 场景2：产品名无法确认（应跳过全部检索，直接输出）----------
    session_id2 = f"query_test_branch_{int(time.time())}"
    logger.info("")
    logger.info(f"[检索图测试] 场景2：产品名无法确认，应跳过检索。session_id={session_id2}")

    # 模拟 node_item_name_confirm 直接产出 answer（反问/拒绝场景）
    # 这里通过给初始状态预置 answer 来触发条件路由的另一条分支
    branch_state = create_query_default_state(
        session_id=session_id2,
        original_query="小米15怎么样？",
        is_stream=False,
        answer="抱歉，未找到相关产品，请提供准确型号以便我为您查询。",
    )
    with usage_context(session_id=session_id2) as acc2:
        query_app.invoke(branch_state)
    done2 = set(get_done_task_list(session_id2))
    logger.info(f"[检索图测试] 场景2 记账：{acc2.text()}")

    logger.info(f"[检索图测试] 场景2 已完成节点：{done2}")

    # 该场景只应执行「生成答案」，不应有任何检索节点
    search_nodes = {
        "切片搜索", "切片搜索(假设性文档)", "查询知识图谱",
        "网络搜索", "倒排融合", "重排序",
    }
    leaked = search_nodes & done2
    if leaked:
        problems.append(f"场景2 不应执行检索节点，却执行了：{leaked}")
    if "生成答案" not in done2:
        problems.append("场景2 未执行答案生成节点")

    # ---------- 汇总 ----------
    logger.info("")
    logger.info("=" * 70)
    if problems:
        for p in problems:
            logger.error(f"[检索图测试] [FAIL] {p}")
        logger.error(f"[检索图测试] 测试失败，共{len(problems)}项未通过")
    else:
        logger.success("[检索图测试] [PASS] 流程结构验证全部通过")
    logger.info("=" * 70)

    # 清理内存态任务记录
    clear_task(session_id)
    clear_task(session_id2)
