import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple


# 命令行直接运行本文件时，将项目根目录加入 sys.path，保证 app 包可导入
if __package__ in (None, ""):
    _this_file = os.path.abspath(__file__)
    _project_root = os.path.abspath(os.path.join(os.path.dirname(_this_file), "..", "..", "..", ".."))
    sys.path.insert(0, _project_root)
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_document_split"

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500


def step1_get_content(state):
    """
    参数校验
    :param state:
    :return:
    """
    function_name = sys._getframe().f_code.co_name
    md_content:str  = state['md_content']
    if not md_content:
        logger.error(f"[{NODE_NAME}] [{function_name}] 没有有效的md内容")
        raise Exception("没有有效的md内容")
    """
    在不同的系统中换行符可能不同
    """
    md_content = md_content.replace('\r\n','\n').replace('\r','\n')
    file_title:str = state.get("file_title","default_title")
    return md_content,file_title


def step2_split_by_title(md_content, file_title):
    title_pattern = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$')
    lines = md_content.split('\n')
    current_title = ""
    current_lines = []
    section_count = 0
    is_code_block = False
    sections = []

    def flush_section(title, content_lines):
        # 去掉标题行本身后仍有非空内容，才保留该段落；否则视为无内容标题，丢弃
        body = '\n'.join(content_lines[1:]).strip()
        if body:
            sections.append({
                "title": title,
                "content": '\n'.join(content_lines),
                "file_title": file_title,
            })
            return True
        return False

    for line in lines:
        strip_line = line.strip()
        if strip_line.startswith('```') or strip_line.startswith('~~~'):
            is_code_block = not is_code_block
            current_lines.append(line)
            continue
        is_title = (not is_code_block) and re.match(title_pattern, strip_line)
        if is_title:
            if current_title:
                section_count += flush_section(current_title, current_lines)
            current_title = strip_line
            current_lines = [current_title]
        else:
            current_lines.append(line)
    if current_title:
        section_count += flush_section(current_title, current_lines)

    return sections, section_count


def split_long_section(section, max_length):
    """
    将当前内容切割(如果内容长度超过了max_length)
    :param section:
    :param max_length:
    :return:
    """
    function_name = sys._getframe().f_code.co_name
    content = section.get('content')
    if content is None:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 段落缺少 content 字段，跳过切割: {section}")
        return [section]
    if len(content) <= max_length:
        logger.debug(f"[{NODE_NAME}] [{function_name}] 当前内容长度 {len(section.get('content'))} 小于最大长度 {max_length}")
        return [section]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,
        separators=['\n\n', '\n', ' ', '。', '，', '']
    )
    sub_sections = []
    # enumerate(iterable)同时获取索引和元素
    for index, chunk in enumerate(splitter.split_text(content),start=1):
        text = chunk.strip()
        title = f"{section.get('title')}_{index}"
        parent_title = section.get('title')
        file_title = section.get('file_title')
        part = index
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "part": part,
            "parent_title": parent_title
        })
    return sub_sections


def merge_short_sections(final_sections, min_length, max_length):
    """
    双指针算法合并短段落
    :param final_sections:
    :param min_length: 短切片阈值，低于该长度的切片尝试并入前一段
    :param max_length: 单切片长度上限，合并后不得超过，否则嵌入时会被静默截断
    :return:
    """
    merge_sections = []
    pre_section = None

    for section in final_sections:
        if pre_section is None:
            pre_section = section
            continue
        # 当前段过短且与前一/后一段同父标题时，并入 pre_section
        is_current_short = len(section.get('content')) < min_length
        is_same_parent = section.get('parent_title') == pre_section.get('parent_title')
        # 链式合并会让长度不断累加，必须检查合并后是否超上限：
        # 超上限的切片在向量化时会被 text[:2048] 静默截断，导致尾部内容检索不到
        merged_length = len(pre_section.get('content')) + 1 + len(section.get('content'))
        is_within_limit = merged_length <= max_length
        if is_current_short and is_same_parent and is_within_limit:
            pre_section['content'] += '\n' + section.get('content')
            # part 号保持前一段的最小值，不随合并后移
        else:
            merge_sections.append(pre_section)
            pre_section = section
    if pre_section is not None:
        merge_sections.append(pre_section)
    return merge_sections


def step3_refine_split(sections, max_length,min_length):
    """
    超过MIN_CONTENT_LENGTH 的 chunk要切割
    小于MIN_CONTENT_LENGTH 的 chunk则合并(同一个parent title)
    :param sections:
    :param MIN_CONTENT_LENGTH:
    :return:
    """
    final_sections = []
    for section in sections:
        # 补上 parent_title：无该字段的 section 视为自身为父标题，避免跨标题合并
        section.setdefault('parent_title', section.get('title'))
        sub_section = split_long_section(section,max_length)
        final_sections.extend(sub_section)

    final_sections = merge_short_sections(final_sections, min_length, max_length)

    for section in final_sections:
        section['part'] = section.get('part') or 1
        # 没有parent_title的section，直接用自身为父标题 这个地方容易忽略导致将无关的两部分section合并
        section['parent_title'] = section.get('parent_title') or section.get('title')
    return final_sections


def step_4_backup_chunks(state, sections):
    """将切割完的chunk进行存储"""
    function_name = sys._getframe().f_code.co_name
    # 备份目录按文档隔离：原实现写死 {local_dir}/chunks.json，
    # 而 local_dir 是多个文档共享的，第二篇文档会覆盖第一篇的备份；
    # 且 MD 输入路径不经过 node_pdf_to_md，local_dir 为空时会写到进程CWD。
    # 这里用 md 所在目录（即「文档自己的目录」），天然按文档隔离。
    md_path = state.get('md_path') or ""
    if md_path:
        backup_dir = os.path.dirname(md_path)
    else:
        # 兜底：没有 md_path 时退回 local_dir/file_title，仍保证按文档隔离
        backup_dir = os.path.join(
            state.get('local_dir') or "",
            state.get('file_title') or "unknown_doc",
        )
    os.makedirs(backup_dir, exist_ok=True)
    file_backup_path = os.path.join(backup_dir, "chunks.json")
    with open(file_backup_path, 'w', encoding='utf-8') as f:
        json.dump(
            sections,
            f,
            ensure_ascii=False, # 中文直接原文存储
            indent=4, # 缩进
        )
    logger.info(f"[{NODE_NAME}] [{function_name}] 备份完成：{file_backup_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    function_name = sys._getframe().f_code.co_name
    # 只打印关键字段：state 中包含整篇 md_content，全量打印会刷屏且易触发控制台编码问题
    logger.info(f"[{function_name}] 开始执行，task_id={state.get('task_id')}，md长度={len(state.get('md_content') or '')}")
    add_running_task(state.get("task_id", ""), function_name)

    try:
        # 1.参数校验
        md_content, file_title = step1_get_content(state)
        # 2.粗粒度切割
        sections, section_count = step2_split_by_title(md_content, file_title)
        # 3.特殊场景,一个文档没有标题,我们就给他一个标题
        if section_count == 0:
            logger.warning(f"文档 [{file_title}] 未切分出有效段落，使用「无标题」兜底")
            sections = [
                {
                    "title": "无标题",
                    "content": md_content,
                    "file_title": file_title,
                }
            ]
        # 4.细粒度切割
        sections = step3_refine_split(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)

        # 5.数据备份和chunk属性的修改
        state['chunks'] = sections
        step_4_backup_chunks(state,sections)


    except Exception as e:
        logger.error(f"[{function_name}] 执行失败，错误信息为{e}")
        raise e
    finally:
        logger.info(f"[{function_name}] 执行完毕，产出切片数={len(state.get('chunks') or [])}")
        add_done_task(state.get("task_id", ""), function_name)

    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"[{NODE_NAME}] [__main__] 本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180使用说明书", "hak180使用说明书.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"[{NODE_NAME}] [__main__] 本地测试 - 测试文件不存在：{test_md_path}")
        logger.info(f"[{NODE_NAME}] [__main__] 请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180使用说明书",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info(f"[{NODE_NAME}] [__main__] 开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"[{NODE_NAME}] [__main__] 本地测试完成 - 处理结果状态：{result_state}")
        logger.info(f"[{NODE_NAME}] [__main__] \n=== 开始执行文档切分节点集成测试 ===")

        logger.info(f"[{NODE_NAME}] [__main__] >> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"[{NODE_NAME}] [__main__] ✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
        # 测试成功：最终生成80个有效Chunk....