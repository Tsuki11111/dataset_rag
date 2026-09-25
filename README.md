# 掌柜智库 · 产品文档知识库 RAG

基于 **LangGraph** 的多路检索 RAG 系统，面向**产品使用文档**（说明书、用户手册）的导入与问答。

两条独立的图：
- **导入图**：PDF/MD → 解析 → 读图 → 切分 → 产品主体识别 → 向量化 → 入 Milvus
- **检索图**：产品确认 → 四路并行检索 → RRF 融合 → 重排 → 生成答案

配套两个 FastAPI 服务和两个前端页面，问答侧用 SSE 实时推送检索过程与流式答案。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangGraph |
| 大模型 | 通义千问（OpenAI 兼容接口，`langchain-openai`） |
| 嵌入 | DashScope `text-embedding-v2`（1536 维，**仅稠密向量**） |
| 向量库 | Milvus 2.5（standalone） |
| 对象存储 | MinIO |
| 会话历史 / 去重记录 | MongoDB |
| PDF 解析 | MinerU（云端 API） |
| Web | FastAPI + 原生 HTML/JS（无框架） |
| 包管理 | uv（Python ≥ 3.12） |

---

## 目录结构

```
app/
├── clients/                 # 外部服务客户端
│   ├── milvus_utils.py      # Milvus 单例、稠密检索、产品名写入
│   ├── minio_utils.py       # MinIO 客户端
│   ├── mongo_history_utils.py   # 会话历史读写
│   ├── mongo_dedup_utils.py     # 上传去重指纹
│   └── neo4j_utils.py       # 知识图谱（未使用）
├── conf/                    # 各服务的配置类（读 .env）
├── core/                    # 日志、提示词加载
├── lm/                      # LLM 客户端、嵌入、重排
├── import_process/          # ── 导入链路 ──
│   ├── agent/main_graph.py      # 导入图编排 + 端到端测试
│   ├── agent/nodes/             # 7 个节点
│   ├── agent/create_collections.py  # Milvus 建表
│   ├── api/file_import_service.py   # 上传服务（含去重、撤回）
│   └── page/import.html         # 上传页面
├── query_process/           # ── 检索链路 ──
│   ├── agent/main_graph.py      # 检索图编排 + 流程测试
│   ├── agent/nodes/             # 8 个节点
│   ├── api/query_service.py     # 查询服务（SSE + 历史接口）
│   └── page/chat.html           # 问答页面
└── utils/                   # 任务追踪、SSE、哈希、文档管理等
prompts/                     # 提示词模板（.prompt）
doc/                         # 待导入的原始 PDF
output/                      # 导入过程的中间产物（按文档隔离）
docker/milvus-compose.yml    # Milvus standalone 编排
```

---

## 进度总览

### 导入链路 —— 已完成 ✅

7 个节点全部实现并端到端验证通过（实测导入 371 切片的 PDF 耗时约 350 秒）。

| 节点 | 状态 | 说明 |
|---|---|---|
| `node_entry` | ✅ | 按后缀路由 PDF/MD，提取 file_title |
| `node_pdf_to_md` | ✅ | MinerU 云端解析，下载解压 md |
| `node_md_img` | ✅ | 图片上传 MinIO + 视觉模型生成描述，替换 md 中的图片链接 |
| `node_document_split` | ✅ | 按 Markdown 标题层级切分，长切片二次切分、短切片合并 |
| `node_item_name_recognition` | ✅ | 大模型识别产品主体 → 写 `kb_item_names` + 生成向量 |
| `node_dashscope_embedding` | ✅ | 批量生成切片稠密向量 |
| `node_import_milvus` | ✅ | 校验 → 幂等清理 → 批量插入 `kb_chunks` → 回填 chunk_id |

**导入服务**（`file_import_service.py`，端口 **8001**）已完成：

- `POST /upload` —— 上传 + **SHA-256 内容去重**（改名仍能识别）+ `force=true` 强制重导
- `GET /documents` —— 已导入文档列表（以 Milvus 为主聚合）
- `DELETE /documents/{file_title}` —— **撤回**（清 Milvus 切片与产品名、去重记录、本地产物、MinIO 对象），带二次确认
- 前端 [import.html](app/import_process/page/import.html)：拖拽上传、重复提示、撤回交互、昼夜模式

### 检索链路 —— 部分完成 🚧

| 节点 | 状态 | 说明 |
|---|---|---|
| `node_item_name_confirm` | ✅ | 7 步完整：LLM 提取产品名+改写问题 → 向量对齐 → 三分支（确认/反问/拒识） → 写历史 |
| `node_search_embedding` | ⬜ | 骨架，待接 `dense_search` |
| `node_search_embedding_hyde` | ⬜ | 骨架，待接 HyDE |
| `node_web_search_mcp` | ⬜ | 骨架，依赖百炼 MCP 联网搜索 |
| `node_query_kg` | ⬜ | 骨架，依赖 Neo4j（本地未部署） |
| `node_rrf` | ⬜ | 骨架，待接 RRF 融合算法 |
| `node_rerank` | ⬜ | 骨架，待接 BGE Reranker |
| `node_answer_output` | 🚧 | SSE 流式推送与历史存档已完成；**答案内容仍是占位文本**，未接 LLM 生成 |

**查询服务**（`query_service.py`，端口 **8002**）已完成：

- `POST /query` —— 提交问题（流式返回 session_id / 非流式直接返回答案）
- `GET /stream/{session_id}` —— **SSE** 推送 `ready` / `progress` / `delta` / `final` / `error`
- `GET /history/{session_id}`、`DELETE /history/{session_id}` —— 会话历史查询与清空
- 前端 [chat.html](app/query_process/page/chat.html)：检索管线可视化、流式答案、昼夜模式

### 未开始 ⬜

- 检索链路的真实检索与答案生成（当前是骨架 + 占位文本）
- Neo4j 知识图谱（`neo4j_utils.py` 无引用，图节点为骨架）
- 重排序（`reranker_utils.py` 无引用）

---

## 快速开始

### 1. 配置 `.env`

```ini
# ── 大模型（通义千问，OpenAI 兼容）──
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-plus
LLM_DEFAULT_TEMPERATURE=0.1
VL_MODEL=qwen3-vl-flash

# ── 嵌入模型（DashScope）──
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v2
EMBEDDING_DIM=1536
EMBEDDING_BATCH_SIZE=16

# ── Milvus ──
MILVUS_URL=http://127.0.0.1:19530
CHUNKS_COLLECTION=kb_chunks
ITEM_NAME_COLLECTION=kb_item_names
MILVUS_METRIC_TYPE=COSINE

# ── MongoDB ──
MONGO_URL=mongodb://127.0.0.1:27017
MONGO_DB_NAME=kb002

# ── MinIO ──
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base-files
MINIO_IMG_DIR=/upload-images

# ── MinerU（PDF 解析）──
MINERU_API_TOKEN=sk-xxx
MINERU_BASE_URL=https://mineru.net/api/v4
```

### 2. 启动依赖服务（Docker）

```bash
# Milvus standalone（etcd + milvus，复用已有的 minio）
docker compose -f docker/milvus-compose.yml up -d

# MongoDB
docker run -d --name mongo -p 27017:27017 -v mongo-data:/data/db mongo:8
```

> 容器均未设 restart policy，Docker Desktop 重启后需手动拉起：
> `docker start minio milvus-etcd milvus-standalone attu mongo mongo-express`

### 3. 建 Milvus 集合（首次）

```bash
.venv/Scripts/python.exe -m app.import_process.agent.create_collections
```

### 4. 启动服务

```bash
# 导入服务 → http://127.0.0.1:8001/import.html
.venv/Scripts/python.exe -m app.import_process.api.file_import_service

# 查询服务 → http://127.0.0.1:8002/chat.html
.venv/Scripts/python.exe -m app.query_process.api.query_service
```

### 5. 命令行跑图（调试用）

```bash
# 导入图端到端测试（改 main_graph.py 里的 TEST_PDF_NAME）
.venv/Scripts/python.exe -m app.import_process.agent.main_graph

# 检索图结构测试（验证分叉/合并/条件路由）
.venv/Scripts/python.exe -m app.query_process.agent.main_graph

# 单节点测试
.venv/Scripts/python.exe -m app.query_process.agent.nodes.node_item_name_confirm
```

---

## 端口一览

| 端口 | 服务 |
|---|---|
| 8000 | Attu（Milvus 图形界面） |
| 8001 | 文档导入服务 |
| 8002 | 知识库查询服务 |
| 8081 | mongo-express（MongoDB 图形界面） |
| 9000 / 9001 | MinIO API / 控制台 |
| 19530 | Milvus |
| 27017 | MongoDB |

---

## 数据模型

**`kb_chunks`**（文档切片）

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | INT64 | 主键，自增 |
| `content` / `title` / `parent_title` | VARCHAR | 切片正文与层级 |
| `part` | INT8 | 长段落二次切分的序号 |
| `file_title` | VARCHAR | 源文档名，**幂等清理依据** |
| `item_name` | VARCHAR | 所属产品主体 |
| `dense_vector` | FLOAT_VECTOR(1536) | 稠密向量，HNSW + COSINE |

**`kb_item_names`**（产品主体）

| 字段 | 类型 | 说明 |
|---|---|---|
| `item_name` | VARCHAR(512) | 主键 |
| `file_title` | VARCHAR | 来源文档 |
| `dense_vector` | FLOAT_VECTOR(1536) | 产品名向量 |

**MongoDB**

| 集合 | 用途 |
|---|---|
| `chat_message` | 会话历史（`session_id` + `ts` 复合索引） |
| `imported_documents` | 上传去重指纹（`file_hash` 唯一索引） |

---

## 设计取舍

**去除稀疏向量**
项目原设计用 BGE-M3 生成稠密+稀疏双向量做混合检索。现改用 DashScope `text-embedding-v2`，**只输出稠密向量**，因此：
- `kb_chunks` / `kb_item_names` 均无 `sparse_vector` 字段
- 检索走 `dense_search` 单路，不再有 `hybrid_search`
- 代价：失去关键词精确匹配能力（型号、参数名这类查询受影响）

**产品名用向量对齐，而非字符串匹配**
用户可能说「HAK180」而库里存的是「Brother HAK 180 烫金机」，字符串匹配无法处理。改用向量相似度分档：
- `≥0.85` 自动确认
- `0.6 ~ 0.85` 反问用户选择
- `<0.6` 判为未找到

**文档身份用「文件内容哈希」**
同一份文件改名后仍应识别为重复，因此去重依据是 SHA-256 而不是文件名。

**幂等策略**
- `kb_chunks`：按 `file_title` 先删后插（**不能按 `item_name`**，不同文档可能描述同一产品，会误删）
- `kb_item_names`：`item_name` 作主键 + 按 `file_title` 清理同文档旧名

---

## 已知问题 / 待办

| 项 | 说明 |
|---|---|
| 检索节点为骨架 | `node_search_embedding` 等 6 个节点只有 `sleep` + 返回空列表，走完分支 A 后拿不到真实结果 |
| 答案仍是占位文本 | `node_answer_output` 在 `state['answer']` 为空时输出固定的演示文本，未接 LLM 生成 |
| `get_recent_messages` 曾取错数据 | 原实现 `sort(ASCENDING).limit(N)` 取的是**最旧** N 条，已修为倒序取再反转为正序 |
| 相似度阈值 | `kb_item_names` 的 0.85/0.6 阈值取自教程代码（教程正文写的是 0.95，两处不一致） |
| 无引用的模块 | `neo4j_utils.py`、`reranker_utils.py`、`format_utils.py`、`mongo_history_utils_new.py` 均无引用 |
