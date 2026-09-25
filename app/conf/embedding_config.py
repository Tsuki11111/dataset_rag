# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（保持和原代码一致，只需执行一次）
load_dotenv()


# 定义Embedding配置（已由本地BGE-M3切换为DashScope text-embedding-v2，与embedding_utils.py保持一致）
@dataclass
class EmbeddingConfig:
    base_url: str    # DashScope OpenAI兼容端点
    api_key: str     # API密钥
    model: str       # 嵌入模型名
    dimension: int   # 向量维度，需与Milvus集合schema一致
    batch_size: int  # 批量请求条数


# 实例化配置对象，和原代码lm_config风格保持一致
embedding_config = EmbeddingConfig(
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    # 复用大模型的密钥，未单独配置EMBEDDING_API_KEY时回退到OPENAI_API_KEY
    api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL"),
    dimension=int(os.getenv("EMBEDDING_DIM") or 1536),
    batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE") or 16)
)
