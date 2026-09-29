# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（保持和原代码一致，只需执行一次）
load_dotenv()

# 默认端点：DashScope 原生重排接口，注意它不是 OpenAI 兼容端点
DEFAULT_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


# 定义重排序配置（已由本地 BGE 切换为 DashScope gte-rerank-v2，与 reranker_utils.py 一致）
@dataclass
class RerankerConfig:
    base_url: str  # DashScope 原生重排端点
    api_key: str   # API密钥
    model: str     # 重排模型名


# 实例化配置对象，和 embedding_config 风格保持一致
reranker_config = RerankerConfig(
    base_url=os.getenv("RERANK_BASE_URL") or DEFAULT_RERANK_URL,
    # 复用大模型的密钥，未单独配置 RERANK_API_KEY 时回退到 OPENAI_API_KEY
    api_key=os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    model=os.getenv("RERANK_MODEL") or "gte-rerank-v2",
)
