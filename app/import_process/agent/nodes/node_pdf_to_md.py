import os
import sys
import time
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

# 节点名，与 main_graph.py 中 add_node 注册的名称保持一致，用于日志前缀
NODE_NAME = "node_pdf_to_md"


def step1_vaildate_paths(state):
    """
    进行路径校验,local_file_path失效直接抛出异常,loacl_dir没有就给默认值
    :param state:
    :return:
    """
    function_name = sys._getframe().f_code.co_name
    logger.debug(f"[{NODE_NAME}] [{function_name}] 在文件转pdf下，开始进行文件路径校验")
    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]
    if not pdf_path:
        logger.error(f"[{NODE_NAME}] [{function_name}] local_file_path为空，无法进行文件路径校验")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        logger.info(f"[{NODE_NAME}] [{function_name}] local_dir没有给默认值,默认值为: {local_dir}")
    from pathlib import Path
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        logger.error(f"[{NODE_NAME}] [{function_name}] pdf_path_obj不存在,无法进行文件路径校验")
        raise ValueError(f"[{NODE_NAME}] [{function_name}] local_dir不存在,无法进行文件路径校验")
    if not local_dir_obj.exists():
        logger.error(f"[{NODE_NAME}] [{function_name}] local_dir_obj不存在,主动创建对应文件夹")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj



def step2_upload_and_poll(pdf_path_obj):
    function_name = sys._getframe().f_code.co_name
    # 1.申请

    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}", "data_id": "abcd"}
        ],
        "model_version": "vlm"
    }
    file_path = ["demo.pdf"]

    response = requests.post(url, headers=header, json=data)
    if response.status_code != 200 or response.json()['code']!= 0:
        logger.error(f"[{NODE_NAME}] [{function_name}] 申请文件上传失败,错误信息: {response.text}")
        raise RuntimeError(f"[{NODE_NAME}] [{function_name}] 申请文件上传失败,错误信息: {response.text}")
    batch_id = response.json()["data"]["batch_id"]
    upload_urls = response.json()["data"]["file_urls"]
    # 2.上传 使用put请求，但是也不能直接发put请求，因为使用了代理之后可能会报错

    http_session = requests.Session()
    http_session.trust_env = False
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        # 预签名URL是按「无自定义请求头」生成的，不能带Content-Type/Authorization，否则OSS签名校验失败
        upload_response = http_session.put(upload_urls[0], data=file_data)
        if upload_response.status_code != 200:
            logger.error(f"[{NODE_NAME}] [{function_name}] 上传文件失败,错误信息: {upload_response.text}")
            raise RuntimeError(f"[{NODE_NAME}] [{function_name}] 上传文件失败,错误信息: {upload_response.text}")
    except Exception as e:
        logger.error(f"[{NODE_NAME}] [{function_name}] 上传文件失败,错误信息: {str(e)}")
        raise e
    finally:
        http_session.close()

    # 3.轮询
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    timeout_seconds = 600
    poll_interval_seconds = 3 # 间隔时间
    start_time = time.time()
    while True:
        # 3.1.超时判断
        if time.time() - start_time > timeout_seconds:
            logger.error(f"[{NODE_NAME}] [{function_name}] MinerU解析超时,无法获取解析结果")
            raise TimeoutError(f"[{NODE_NAME}] [{function_name}] MinerU解析超时,无法获取解析结果")
        # 3.2.向指定的url获取本次解析的结果
        res = requests.get(url, headers=header)
        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                time.sleep(poll_interval_seconds)
                continue
            raise RuntimeError(f"[{NODE_NAME}] [{function_name}] MinerU获取解析结果失败,错误信息: {res.text}")
        json_data = res.json()
        if json_data['code'] != 0:
            logger.error(f"[{NODE_NAME}] [{function_name}] MinerU解析失败,错误信息: {json_data['message']}")
            raise RuntimeError(f"[{NODE_NAME}] [{function_name}] MinerU解析失败,错误信息: {json_data['message']}")
        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            full_zip_url = extract_result['full_zip_url']
            logger.info(f"[{NODE_NAME}] [{function_name}] 已经完成pdf解析，耗时: {time.time() - start_time}秒，解析结果: {full_zip_url}")
            return full_zip_url



def step3_download_and_extract(zip_url, local_dir_obj, pdf_name):
    """
    下载zip包,并且解压,返回解压后的md文件路径
    :param zip_url:
    :param lode_dir_obj:
    :param pdf_name:
    :return: md_path
    """
    function_name = sys._getframe().f_code.co_name
    # 1.下载zip包 response响应体
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[{NODE_NAME}] [{function_name}] 下载zip包失败,错误信息: {response.text}")
        raise RuntimeError(f"[{NODE_NAME}] [{function_name}] 下载zip包失败,错误信息: {response.text}")
    # 2.将zip文件保存到本地
    zip_save_path = local_dir_obj / f"{pdf_name}_result.zip"
    with open(zip_save_path, 'wb') as f:
        f.write(response.content)
        logger.info(f"[{NODE_NAME}] [{function_name}] 下载zip包成功,保存路径: {zip_save_path}")
    # 3.清空一下旧目录(将上一次处理的文件目录删除)
    extract_target_dir = local_dir_obj / pdf_name
    # 两次解压出的文件可能不一样,文件夹里无法完全覆盖,所以先清空一下旧目录
    if extract_target_dir.exists():
        import shutil
        shutil.rmtree(extract_target_dir)
        logger.info(f"[{NODE_NAME}] [{function_name}] 清空旧目录: {extract_target_dir}")
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    # 4.解压zip包
    import zipfile
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)
    # 5.返回解压后的md文件路径
    # 解压后的文件名可能叫文件.md或者full.md
    md_file_list = list(extract_target_dir.rglob('*.md'))

    if not md_file_list:
        logger.error(f"[{NODE_NAME}] [{function_name}] 解压后的md文件列表为空,无法进行解析")
        raise RuntimeError(f"[{NODE_NAME}] [{function_name}] 解压后的md文件列表为空,无法进行解析")

    target_md_file: Path | None = None

    for md in md_file_list:
        if md.name == pdf_name + '.md':
            target_md_file = md
            break
    if not target_md_file:
        for md in md_file_list:
            # islower() 返回 bool，原来写成 md.name.islower() == 'full.md' 恒为False，是死代码
            if md.name.lower() == 'full.md':
                target_md_file = md
                break
    if not target_md_file:
        target_md_file = md_file_list[0]
    if target_md_file.stem != pdf_name:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{pdf_name}.md"))
        logger.info(f"[{NODE_NAME}] [{function_name}] 解压后的md文件名不正确,已重命名为: {target_md_file}")
    logger.info(f"[{NODE_NAME}] [{function_name}] 解压成功!解压后的md文件路径: {target_md_file}")
    return str(target_md_file)


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1.进入的日志和任务状态的配置
    2.进行参数校验 local_dir 给予默认值 local_file_path 完成字面意思的校验 --> 深入校验校验的文件是否真实存在
    3.调用MinerU进行PDF的解析 local_file_path 返回一个下载文件的地址 xx.zip url地址
    4.下载zip包,并且解析和提取 local_dir
    5.把md_path地址进行赋值,读取md的文件内容md_content 赋值2
    6.结束的日志任务状态的配置
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"[{function_name}] 节点开始执行,现在状态为: {state['task_id']}")
    add_running_task(state["task_id"], function_name)
    try:
        # 校验后的文件和输出文件夹Path对象
        pdf_path_obj,local_dir_obj = step1_vaildate_paths(state)

        zip_url = step2_upload_and_poll(pdf_path_obj)

        md_path = step3_download_and_extract(zip_url, local_dir_obj,pdf_path_obj.stem)

        state["md_path"] = md_path
        state["local_dir"] = local_dir_obj
        with open(md_path, 'r', encoding='utf-8') as f:
            state["md_content"] = f.read()
    except Exception as e:
        logger.error(f"[{function_name}] 使用MinerU进行PDF的解析时发生错误,错误信息: {str(e)}")
        raise e
    finally:
        logger.info(f"[{function_name}] 节点执行结束,现在状态为: {state['task_id']}")
        add_done_task(state["task_id"], function_name)
    return state
if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info(f"[{NODE_NAME}] [__main__] ===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"[{NODE_NAME}] [__main__] 测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180使用说明书.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info(f"[{NODE_NAME}] [__main__] ===== 结束node_pdf_to_md节点单元测试 =====")