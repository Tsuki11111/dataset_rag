import os
import re
import sys
import base64
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_md_img"

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step1_get_content(state) -> Tuple[str, Path, Path]:
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("Markdown 文件路径不存在")
    md_file_path_obj = Path(md_file_path)
    if not md_file_path_obj.exists():
        raise ValueError("Markdown 文件路径不存在")
    # md_content 可能已被上游节点（如node_pdf_to_md）写入，此时直接用状态里的内容
    md_content = state.get('md_content')
    if not md_content:
        with open(md_file_path_obj, 'r', encoding='utf-8') as f:
           md_content = f.read()
        state['md_content'] = md_content
    images_dir_obj = md_file_path_obj.parent/"images"
    # 这个地方写的是固定的，暂时先放一放
    return md_content,md_file_path_obj,images_dir_obj


def find_image_in_md(md_content, image_file, context_len=100):
    """
    在Markdown中查找图片的上下文
    :param md_content: Markdown 全文
    :param image_file: 图片文件名（含后缀）
    :param context_len: 上下文截取的最大字符数
    :return: 找到返回 (上文, 下文)；未找到返回 None
    """
    function_name = sys._getframe().f_code.co_name
    logger.debug(f"[{NODE_NAME}] [{function_name}] 开始查找图片上下文 image_file={image_file}, context_len={context_len}")
    if not md_content or not image_file:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 查找图片上下文参数无效 md_content为空={not md_content}, image_file为空={not image_file}")
        return None

    name = re.escape(image_file)
    # 兼容 Markdown 中常见的 URL 编码形式（如空格被编码为 %20）
    encoded = re.escape(urllib.parse.quote(image_file))
    pattern = re.compile(rf'!\[[^\]]*\]\([^)\s]*?({name}|{encoded})\)')
    match = pattern.search(md_content)
    if not match:
        logger.warning(f"[{NODE_NAME}] [{function_name}] 图片 [{image_file}] 未在 Markdown 中找到引用,上下文为空")
        return None

    start, end = match.span()
    before = md_content[:start]
    after = md_content[end:]

    # 上文：图片所在段落（以空行分隔）中位于图片之前的文本
    before_stripped = before.rstrip()
    para_start = before_stripped.rfind('\n\n')
    if para_start != -1:
        para_start += 2
    else:
        para_start = 0
    context_before = before_stripped[para_start:].strip()[-context_len:]

    # 下文：图片之后紧跟的段落（跳过图片行尾的空行）
    after_stripped = after.lstrip()
    para_end = after_stripped.find('\n\n')
    if para_end == -1:
        para_end = len(after_stripped)
    context_after = after_stripped[:para_end].strip()[:context_len]

    logger.info(
        f"[{NODE_NAME}] [{function_name}] 图片 [{image_file}] 上下文提取成功 上文_len={len(context_before)} "
        f"下文_len={len(context_after)} 上文={context_before!r} 下文={context_after!r}"
    )
    return context_before, context_after


def step2_scan_images(md_content, img_dir) -> List[Tuple[str,str,Tuple[str,str]]]:
    function_name = sys._getframe().f_code.co_name
    targets = []
    for image_file in os.listdir(img_dir):
        if not is_supported_image(image_file):
            logger.warning(f"[{NODE_NAME}] [{function_name}] 当前文件:{image_file},不是图片格式,无需处理")
            continue
        # 如果是图片，就在md中查询,如果查询到,获取其上文和下文即可
        context_data = find_image_in_md(md_content,image_file)

        if not context_data:
            logger.warning(f"[{NODE_NAME}] [{function_name}] 当前文件:{image_file},未在Markdown中找到其使用,上下文为空")
            continue
        targets.append((image_file,str(img_dir/image_file),context_data))
    return targets


def step3_generate_summary(targets, stem):
    """
    利用视觉模型获取图片内容的描述
    :param targets:
    :param stem:
    :return:
    """
    function_name = sys._getframe().f_code.co_name
    summaries = {}
    request_times = deque()
    for image_file, img_path, context_data in targets:
        # 为了快速测试所以先不进行限速
        # apply_api_rate_limit(request_times=request_times, max_requests=10)
        vm_model = get_llm_client(model=lm_config.lv_model)
        # 必须传当前这张图的上下文元组，之前误传了整个 targets 列表，
        # 导致提示词里塞的是元组的字符串形式、且每张图都用同一份，图片描述完全脱离文档语义
        prompt = load_prompt("image_summary",root_folder=stem,image_content=context_data)

        with open (img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8") # 字节转base64字符串


        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # 可以直接放图片的网络访问地址"url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                            # base64图片转化后的字符串
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            },
        ]
        response = vm_model.invoke(messages)
        summary = response.content.strip().replace("\n", "")
        logger.info(f"[{NODE_NAME}] [{function_name}] 图片 [{image_file}] 描述生成成功，描述内容: {summary}")
        summaries[image_file] = summary
    logger.info(f"[{NODE_NAME}] [{function_name}] 图片描述生成成功，共生成 {len(summaries)} 个图片描述")
    return summaries


def _sanitize_alt_text(summary: str) -> str:
    """
    净化视觉模型生成的图片描述，使其可安全用作 Markdown 的 alt 文本

    描述里可能回显上下文中的 Markdown 语法：修好上下文传参后，模型能读到图片
    前后的正文，实测出现过它把邻近的 ![](images/xxx.jpg) 原样抄进描述的情况。
    这样的描述嵌入 ![...](...) 后会形成嵌套的畸形语法，内层本地路径无法再被替换，
    导致该图片残留在最终 md 里。
    :param summary: 视觉模型原始输出
    :return: 清理后的描述；清理后为空则返回通用占位
    """
    if not summary:
        return "图片"
    # 1. 去掉回显的 Markdown 图片/链接语法
    cleaned = re.sub(r'!?\[[^\]]*\]\([^)]*\)', '', summary)
    # 2. alt 文本里不能有 ASCII 方括号/圆括号，否则会破坏 ![...](...) 结构
    #    （中文全角括号（）不受影响，予以保留）
    cleaned = re.sub(r'[\[\]()]', '', cleaned)
    # 3. 去掉换行/制表符，避免破坏单行图片语法
    cleaned = cleaned.replace('\n', '').replace('\r', '').replace('\t', '').strip()
    return cleaned or "图片"


def step4_upload_image_and_replace_md(summaries, md_content, targets, stem):
    """
    1.将图片上传到minio服务器
    2.替换原md中对图片的描述
    :param summaries:
    :param md_content:
    :param targets:
    :param stem:
    :return: 新的Markdown内容
    """
    function_name = sys._getframe().f_code.co_name
    minio_client = get_minio_client()
    list_objs = minio_client.list_objects(minio_config.bucket_name, prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",recursive=True)
    delete_obj_list = [DeleteObject(obj.object_name) for obj in list_objs]
    if delete_obj_list:
        del_errors = minio_client.remove_objects(minio_config.bucket_name, delete_obj_list)
        for del_err in del_errors:
            logger.error(f"[{NODE_NAME}] [{function_name}] 删除 MinIO 对象失败，错误信息: {del_err}")
    logger.info(f"[{NODE_NAME}] [{function_name}] 已经完成了对{minio_config.minio_img_dir}下的image清空")
    image_url = {}
    for image_file, img_path, _ in targets:
        # 1.将图片上传到minio服务器
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=img_path ,
                content_type="image/jpeg",
            )
            image_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"[{NODE_NAME}] [{function_name}] 图片 [{image_file}] 已上传到 MinIO,访问地址为{image_url[image_file]}")
        except Exception as e:
            logger.error(f"[{NODE_NAME}] [{function_name}] 图片 [{image_file}] 上传到 MinIO 失败，错误信息: {e}")
    image_infos = {}
    for image_file,summary in summaries.items():
        if url := image_url.get(image_file):
            image_infos[image_file] = (summary,url)
    logger.info(f"[{NODE_NAME}] [{function_name}] 图片描述处理成功，汇总结果：{image_infos}")
    if image_infos:
        for image_file, (summary,url) in image_infos.items():
            rep = re.compile(r"!\[.*?\]\(.*?"+re.escape(image_file)+r".*?\)")
            # 用 lambda 作替换串：summary 是视觉模型的输出，可能带 LaTeX 反斜杠
            # （如 \textregistered），直接当替换串会被 re.sub 当转义解析并抛 bad escape
            safe_summary = _sanitize_alt_text(summary)
            replacement = f"![{safe_summary}]({url})"
            md_content = rep.sub(lambda _m: replacement, md_content)
    logger.info(f"[{NODE_NAME}] [{function_name}] Markdown内容替换成功，新的Markdown内容为: {md_content}")
    return md_content

def step5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md的备份,并返回新的地址
    :param new_md_content:
    :param md_path_obj:
    :return:
    """
    function_name = sys._getframe().f_code.co_name
    # 用原始文件名推导，避免重复执行时在已带 _new 的路径上再拼一次（产生 X_new_new.md）
    base = os.path.splitext(str(md_path_obj))[0]
    if base.endswith("_new"):
        base = base[:-len("_new")]
    new_md_path = base + "_new.md"
    with open(new_md_path, 'w', encoding='utf-8') as f:
        f.write(new_md_content)
    logger.info(f"[{NODE_NAME}] [{function_name}] 新的Markdown文件已保存到 {new_md_path}")
    return new_md_path
    pass
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{function_name}] 节点开始执行,现在状态为{state}")
    add_running_task(state.get("task_id", ""), function_name)
    try:
        # 1.校验并获取本次操作的数据
        md_content,md_path_obj,img_dir = step1_get_content(state)
        if not img_dir.exists():
            logger.warning(f"图片目录 {img_dir} 不存在，跳过图片处理步骤")
            return state
        # 2.扫描md中的图片获取上下文
        targets = step2_scan_images(md_content,img_dir)
        # 3.通过视觉理解模型获得图片的描述总结
        summaries = step3_generate_summary(targets,md_path_obj.stem)
        # 4.上传图片到minio同时替换md中的图片
        new_md_content = step4_upload_image_and_replace_md(summaries,md_content,targets,md_path_obj.stem)
        # 5.保存新的Markdown内容并获取新的地址
        new_md_file_path = step5_replace_md_and_save(new_md_content,md_path_obj)
        state['md_path'] = new_md_file_path
        state['md_content'] = new_md_content
        logger.info(f"[{function_name}] 节点执行完成,新的Markdown文件路径为{new_md_file_path}")
    finally:
        add_done_task(state.get("task_id", ""), function_name)
    return state
if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
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
            "md_content": ""
        }
        logger.info(f"[{NODE_NAME}] [__main__] 开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
