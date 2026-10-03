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
| 重排 | DashScope `gte-rerank-v2`（**原生端点，非 OpenAI 兼容**） |
| 向量库 | Milvus 2.5（standalone） |
| 对象存储 | MinIO |
| 会话历史 / 去重记录 | MongoDB |
| 知识图谱 | Neo4j 5.26（Community，本地 Docker） |
| 图谱可视化 | ECharts 5.4.3（**CDN 引入**，三源回退 + 文本降级） |
| PDF 解析 | MinerU（云端 API） |
| 联网搜索 | 百炼 MCP `EnhancedSearch`（openai-agents 客户端，**`mcp<2`**） |
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
│   ├── mongo_usage_utils.py     # 调用账本（每次模型调用的 tokens/延迟/成本）+ 报表
│   ├── mcp_search_utils.py  # 百炼 MCP 联网搜索（Streamable HTTP）
│   └── neo4j_utils.py       # Neo4j 知识图谱读写、幂等清理、图谱统计
├── conf/                    # 各服务的配置类（读 .env），含 pricing_config.py（模型计价表）
├── core/                    # 日志（人读文本 + 机器读 JSONL + log_query 查询入口）、提示词加载、
│                            #   request_context（请求归因）、usage_tracker（单次调用记账）
├── lm/                      # LLM 客户端、嵌入、重排
├── import_process/          # ── 导入链路 ──
│   ├── agent/main_graph.py      # 导入图编排 + 端到端测试
│   ├── agent/nodes/             # 8 个节点
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

8 个节点全部实现并端到端验证通过（实测导入 371 切片的 PDF 耗时约 350 秒）。

| 节点 | 状态 | 说明 |
|---|---|---|
| `node_entry` | ✅ | 按后缀路由 PDF/MD，提取 file_title |
| `node_pdf_to_md` | ✅ | MinerU 云端解析，下载解压 md |
| `node_md_img` | ✅ | 图片上传 MinIO + 视觉模型生成描述，替换 md 中的图片链接 |
| `node_document_split` | ✅ | 按 Markdown 标题层级切分，长切片二次切分、短切片合并 |
| `node_item_name_recognition` | ✅ | 大模型识别产品主体 → 写 `kb_item_names` + 生成向量 |
| `node_dashscope_embedding` | ✅ | 批量生成切片稠密向量 |
| `node_import_milvus` | ✅ | 校验 → 幂等清理 → 批量插入 `kb_chunks` → 回填 chunk_id |
| `node_import_kg` | ✅ | 切片分批交 LLM 抽实体+关系 → 写入 Neo4j（含切片挂载）；Neo4j 不可用时跳过不中断导入 |

**导入服务**（`file_import_service.py`，端口 **8001**）已完成：

- `POST /login` / `POST /logout` —— 用访问密钥换 / 清 HttpOnly 会话 Cookie
- `POST /upload` —— 上传 + **SHA-256 内容去重**（改名仍能识别）+ `force=true` 强制重导
- `GET /documents` —— 已导入文档列表（以 Milvus 为主聚合，附图谱实体数）
- `GET /documents/graph?file_title=` —— 单文档知识图谱（实体 + 关系，供前端可视化）
- `DELETE /documents/{file_title}` —— **撤回**（清 Milvus 切片与产品名、Neo4j 图谱、去重记录、本地产物、MinIO 对象），带二次确认
- 前端 [import.html](app/import_process/page/import.html)：拖拽上传、重复提示、撤回交互、昼夜模式、**知识图谱可视化**（点文档行的「图谱」按钮，力导向图展示实体关系，支持按类型筛选与搜索）

### 检索链路 —— 部分完成 🚧

8 个节点全部完成 ✅

| 节点 | 状态 | 说明 |
|---|---|---|
| `node_item_name_confirm` | ✅ | 7 步完整：LLM 提取产品名+改写问题 → 向量对齐 → 三分支（确认/反问/拒识） → 写历史 |
| `node_search_embedding` | ✅ | 改写问题 → 向量化 → `dense_search`（带 `item_name` 过滤）→ `embedding_chunks`；单节点实测 Top1 0.64 |
| `node_search_embedding_hyde` | ✅ | LLM 生成假设文档 → 「问题+假设文档」向量化 → 检索 → `hyde_embedding_chunks` + `hyde_doc`；单节点实测 Top1 0.71 |
| `node_web_search_mcp` | ✅ | 异步调百炼 MCP 增强搜索（工具 `search_pro`）→ `web_search_docs`；图内实测返回 5 条 |
| `node_query_kg` | ✅ | 图谱检索：问题里的实体 → 种子 + 一跳邻居 → 取回切片（正文回 Milvus 取）；图内实测召回 5 条，4 条进 RRF |
| `node_rrf` | ✅ | 加权 RRF 融合切片类召回（基线 / HyDE / 图谱，k=60）→ `rrf_chunks`；图内实测 5+5 输入去重融合为 6 条 |
| `node_rerank` | ✅ | 合并本地切片 + 联网结果为统一格式 → DashScope 重排打分 → 动态 Top-K（断崖截断）→ `reranked_docs`；图内实测 6+5 输入输出 8 条 |
| `node_answer_output` | ✅ | 用 `reranked_docs` 组装上下文调 LLM 生成答案；流式逐块推送，解析【图片】区块作为配图；答案存档 |

**查询服务**（`query_service.py`，端口 **8002**）已完成：

- `POST /login` / `POST /logout` —— 用访问密钥换 / 清 HttpOnly 会话 Cookie
- `POST /query` —— 提交问题（流式返回 session_id / 非流式直接返回 `answer` + `images` + `usage`）
- `GET /stream/{session_id}` —— **SSE** 推送 `ready` / `progress` / `delta` / **`usage`** / `final` / `error`
  （`final` 带 `answer` 与 `images`；`usage` 每记完一笔就推一次累计用量，见[「调用记账」](#调用记账)）
- `GET /history/{session_id}`、`DELETE /history/{session_id}` —— 会话历史查询与清空
- 前端 [chat.html](app/query_process/page/chat.html)：检索管线可视化、流式答案（**Markdown 渲染**）、**答案配图**、昼夜模式

### 未开始 ⬜

功能节点暂无待办——导入链路与检索链路共 16 个节点均已实现并验证。

**企业化改造**（可观测性、Durable Execution、多租户、治理）的路线与逐项进度见下面
「企业化改造路线」一节。

零散的后续方向：给图谱侧排序做更可靠的权重、把其余 4 份文档重新导入以生成图谱、答案配图做去重与缩略图。

---

## 企业化改造路线（待办与进度）

参考 [《Zero2Agent · Agent Infra》](https://onefly.top/zero2Agent/learn-agent-basic/09-agent-infra/index.html)
的六层标准。**现状**：Harness 层已相当完整（16 个节点、四路召回、图谱、MCP），
但 Infra 层几乎是空的——对照文中递进（Demo → 内部工具 → **面向用户产品** → 企业平台 → 强合规），
目前处在中间偏下。

> **本节是路线的唯一权威**，HANDOFF 只留指针，避免两份文档各自演化。
> 当前进度：**16 项完成 3 项**。

文中两个判断值得记住：
「**Checkpoint + Durable Execution 是从 Demo 到生产最关键一步**」，
以及「多租户、成本归因、合规审计、Guardrails、灰度占据企业 Infra 70% 以上工作量」。

### 现在就存在的三个问题

这三条是后面所有工作的动机，也是判断优先级时的依据。

| 问题 | 现状 |
|---|---|
| **异常被 `except` 吞掉** | 实现 `node_query_kg` 时一个 `NameError` 被「失败不中断链路」的兜底 `except` 降级成 warning，图谱那一路静默返回空、功能等于废了，只有测试断言才发现 |
| **任务状态在内存里** | `task_utils` 用普通 dict 存运行/完成列表，**服务一重启，进行中的导入就凭空消失** |
| ~~**成本完全不可见**~~ | **✅ 已解决**，见[「调用记账」](#调用记账)：一次问答要调 LLM + 四路召回 + 重排 + 图谱 + MCP，此前花了多少、谁花的账上一片空白；现在每次调用逐笔入账，可按类型 / 模型 / 租户 / 单次请求归集 |

### Phase 1 · 地基（最该先做）

- [x] **访问鉴权与租户标识** —— ✅ 已完成，见[「访问鉴权」](#访问鉴权)一节：
      API Key + HttpOnly Cookie，9 个数据接口全部受保护。
      **但只做认证、不做数据隔离**：密钥带 `tenant_id`，数据尚未按它过滤
- [x] **单次调用记账** —— ✅ 已完成，见[「调用记账」](#调用记账)一节：
      LangChain callback 采集 LLM 用量 + 三处手工埋点（嵌入 / 重排 / 联网搜索），
      逐笔写 MongoDB `llm_usage`，可按类型 / 模型 / 租户 / 单次请求出报表。
      `tenant_id` 已经进账本，为 Phase 3 的成本归因留好了接口
- [x] **结构化日志** —— ✅ 已完成，见[「结构化日志」](#结构化日志)一节：
      控制台保留人读格式并挂上 `[trace·node]` 标签，另存一份 JSONL 给机器查；
      归因字段复用记账那套 `request_context`，业务代码一行没改
- [ ] **异常分级处置** —— 按 timeout / 429 / 5xx / refusal / invalid tool args 分别处理，
      替掉现在一律 `except: return []` 的写法

### Phase 2 · 可靠性

- [ ] **LangGraph checkpointer**（`SqliteSaver` / `PostgresSaver`）
      —— 框架自带，接上即可从断点续跑；文中称这是「从 Demo 到生产最关键一步」
- [ ] 故障分类重试：timeout 有限重试、429 退避、refusal 不盲重试、invalid tool args 绝不执行
- [ ] 单节点超时 + 整个查询的 wall-clock / token 预算
- [ ] `task_utils` 从内存搬到 Redis / Postgres

### Phase 3 · 多租户与观测

- [ ] 四个存储加 `tenant_id`：Milvus（**要重建集合**）、Neo4j、MongoDB、MinIO 路径前缀
- [ ] per-tenant 并发槽位、token 预算、工具频率限制
- [ ] 分布式 trace（OpenTelemetry / Langfuse）
- [ ] 成本归因与分摊

### Phase 4 · 治理

- [ ] Guardrails：输入护栏、输出护栏、工具护栏
- [ ] append-only 审计日志（`who` / `what` / `risk_score` / `approver`）
- [ ] 高危操作人工审批 —— 「撤回文档」就是典型场景
- [ ] Agent 版本管理（prompt hash + tool set hash + model version）与灰度发布

### 优先级建议

**别按顺序全做。** 文中那张递进表本质是在说：做到「面向用户产品」那一档就已经拿到 80% 的收益，
多租户 + chargeback + 灰度那一档是给多团队平台准备的，单人 / 小团队做进去性价比很低。

**Phase 1 的前两项无论如何先做**——它们不依赖任何架构决策，而且做完之后才有数据判断后续该往哪投。

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

# ── 重排模型（DashScope，注意不是 OpenAI 兼容端点）──
# 不设置时复用 OPENAI_API_KEY
RERANK_MODEL=gte-rerank-v2

# ── Milvus ──
MILVUS_URL=http://127.0.0.1:19530
CHUNKS_COLLECTION=kb_chunks
ITEM_NAME_COLLECTION=kb_item_names
MILVUS_METRIC_TYPE=COSINE

# ── MongoDB ──
MONGO_URL=mongodb://127.0.0.1:27017
MONGO_DB_NAME=kb002

# ── Neo4j 知识图谱 ──
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_DATABASE=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=123123123

# ── MinIO ──
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base-files
MINIO_IMG_DIR=/upload-images

# ── MinerU（PDF 解析）──
MINERU_API_TOKEN=sk-xxx
MINERU_BASE_URL=https://mineru.net/api/v4

# ── 百炼 MCP（联网搜索，供 node_web_search_mcp 使用）──
# 鉴权复用 OPENAI_API_KEY；Streamable HTTP 协议，服务端无状态（响应不带 session-id）
# 目前仅一个工具 search_pro，参数 query
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1/mcps/EnhancedSearch/mcp
```

### 2. 启动依赖服务（Docker）

```bash
# Milvus standalone（etcd + milvus，复用已有的 minio）
docker compose -f docker/milvus-compose.yml up -d

# MongoDB
docker run -d --name mongo -p 27017:27017 -v mongo-data:/data/db mongo:8

# Neo4j 知识图谱（独立 compose，自带 project 名，不会与 Milvus 的容器互相干扰）
docker compose -f docker/neo4j-compose.yml up -d
```

> 容器均未设 restart policy，Docker Desktop 重启后需手动拉起：
> `docker start minio milvus-etcd milvus-standalone attu mongo mongo-express neo4j`

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

## 访问鉴权

数据接口需要**访问密钥**；页面本身与 `/health` 保持公开。

```bash
# 建第一个用户（密钥只在这一刻打印一次，丢了只能重建）
.venv/Scripts/python.exe -m app.clients.mongo_user_utils add 张三 admin
.venv/Scripts/python.exe -m app.clients.mongo_user_utils list      # 查看
.venv/Scripts/python.exe -m app.clients.mongo_user_utils revoke 张三  # 撤销
```

两种提交方式，任选其一：

- `Authorization: Bearer <key>` —— 给脚本 / 程序化调用
- 浏览器访问任意页面 → 接口返回 401 时自动弹出密钥输入框 → 登录成功后写入 HttpOnly Cookie

**为什么浏览器侧走 Cookie 而不是让前端存密钥发请求头**：查询服务的流式接口用的是 `EventSource`，
而它**无法自定义请求头**——Bearer 头在那条路上根本发不出去。Cookie 由浏览器自动携带，
页面里十几处 `fetch` 与那个 `EventSource` 一行都不用改。

**当前只做「认证」、不做「数据隔离」**：密钥对应一个 `tenant_id`，但数据还没按它隔离
（给 Milvus / Neo4j / MongoDB / MinIO 四个存储加租户维度是停机迁移级别的改动）。
所以**现阶段任何有效密钥都能看到全部数据**。这一层的价值是把身份打通、为后续隔离留好接口。

---

## 调用记账

每一次外部模型调用（LLM 生成 / 嵌入 / 重排 / 联网搜索）都记一笔账，回答「钱花在哪」。

```bash
# 看最近 1 天 / 7 天的账（按类型、模型、租户、最贵的几次请求）
.venv/Scripts/python.exe -m app.clients.mongo_usage_utils
.venv/Scripts/python.exe -m app.clients.mongo_usage_utils 7
```

```python
# 某一次问答的明细（排查「这次怎么这么贵」）
from app.clients.mongo_usage_utils import summarize_trace
summarize_trace("<trace_id>")
```

**页面上也看得到**：每张答案卡片底部有一行「本次消耗」——`↑9.1k ↓1.03k · ¥0.0093 · 8 次调用 · 20.4s`。
流式回答时它**随每次模型调用结束实时跳动**（正在跑时圆点是活的，跑完停跳并补上总耗时），
点「明细」展开还能看到按节点拆分的调用次数 / tokens / 成本——
`rerank` 这类不显眼的环节花掉多少，一眼就能看见。

实现上记账层不碰 SSE：`usage_context(on_usage=...)` 收一个进度回调，
查询服务把「推 `usage` 事件给前端」这件事作为回调传进去，依赖方向保持单向。
非流式路径则直接用 `/query` 响应里的 `usage` 字段。

**首次实测**（一次完整问答，「Brother HAK 180 烫金机怎么安装烫金膜盒？」，
8 次调用、约 1 万 tokens、估算 **0.0095 元**，其中）：

| 环节 | 成本占比 | 说明 |
|---|---|---|
| 生成答案 | 47% | 上下文最大（7 条切片 ≈ 4.7k 字符），且是流式长输出 |
| 重排 | 43% | 15 条候选一起打分，输入 tokens 达 5k |
| HyDE（LLM 生成假设文档 + 嵌入） | 7% | |
| 产品名确认 / 向量检索 | 1.4% | |
| 联网搜索 | 0% | 按次计费、单价未公开，只记次数不计成本 |

单次样本不代表长期分布，但已经能回答一个具体问题：**这条链路里最贵的不是 LLM 生成，
而是「把 15 条候选全喂给重排」**——想降本应从这里下手（比如先粗筛再精排），
而不是去换更便宜的生成模型。

**记账的边界**：MinerU 解析按页数配额计费、与 token 无关，不在账本内。

### 三个设计取舍

**LangChain callback 挂在 LLM 客户端上，而不是各调用点传 handler**
16 个节点里有 8 处模型调用点，逐个传 handler 既侵入又必漏——**漏了还不报错**，
只是账本上少一笔。挂在 `get_llm_client()` 返回的客户端上（它是全局缓存的单例），
一次配置全项目生效，新增节点也自动被覆盖。

**归因走 ContextVar，而不是函数参数**
账本要回答「谁花的」，就得知道这次调用属于哪个租户、哪次会话、哪个节点。这些信息
在服务入口和节点里天然存在，但传到底层 `generate_embeddings()` 要多穿 5 层。
改用 ContextVar 存「当前请求」，入口设一次即可——**关键在于 LangGraph 的并发执行器
提交节点任务前会 `copy_context()`**，所以四路并发检索（各自跑在不同线程）都能读到
同一份归因，实测 8 次调用全部正确落到 `node_rerank` / `node_answer_output` 等节点名下。

节点头上的归因不是每个节点自己写的，而是 `main_graph` 注册时用 `add_tracked_node`
包一层——一张图只有一处需要维护，节点内部一行都不用改。

**账本只把 tokens 当事实，成本是算出来的**
单价会变（qwen-plus 就调过价）。记录里带当时的估算值，但**报表一律按当前计价表重算**，
这样单价更新后历史账目跟着走，而不是冻结在当初的估算上。
计价表在 `app/conf/pricing_config.py`，来源与查询日期逐条写在注释里。

三处刻意的简化都在明面上：qwen-plus 的阶梯计价只取最低档（本项目上下文远小于 128k）；
联网搜索按次计费、单价未公开，只记次数并把成本记为「未计价」而不是 0；
**失败的调用也入账**（`ok=false` + 错误摘要），因为「哪条路经常挂」只有失败记录能回答。

### 两个实现上的坑

**流式必须显式开 `stream_usage`**
`answer_output` 用 `llm.stream()` 生成答案，这是单次最贵的调用。而 langchain-openai
只在 base_url 是 OpenAI 官方地址时才默认开启 `stream_usage`——本项目指向 DashScope，
不显式传 `stream_usage=True` 就拿不到用量，**最贵的那笔账会静默漏记**。
已在 `lm_utils.get_llm_client()` 里显式打开，并实测 DashScope 兼容端点会在最后一个
chunk 返回 usage。

**记账绝不能反过来弄坏主流程**
所有落库与用量解析都吞异常并降级为 warning：一次账单写不进去，不该让用户拿不到答案。
回调里抛异常还会污染 LangChain 的调用链，`TokenUsageCallback` 的两个回调体都整体兜了。

**数据库挂了也不能拖慢业务**：Mongo 连接超时收到 2 秒（pymongo 默认 30 秒），
且失败后熔断 60 秒——否则每次模型调用都要重撞一遍连接超时。
实测库不可用时首笔 2.8s、后续每笔 0.004s（改前每笔 30s）。
熔断期结束会自动重试，不会永久放弃记账。

---

## 结构化日志

**同一条日志记录，三种渲染，各给各的读者**——不会各记各的，因为没有第二份数据源：

| 输出 | 格式 | 给谁 |
|---|---|---|
| 控制台 | 彩色人读，行首挂 `[9b2063cb·node_rerank]` 标签 | 开发时盯着看 |
| `logs/app_年月日.log` | 同上但不带颜色 | `grep` / 翻历史 |
| `logs/app_年月日.jsonl` | 每行一个 JSON 对象 | 程序、`jq`、将来接采集 |

```bash
# 看某一次请求的全部日志（一次问答 = 一个 trace，与账本里的 trace_id 相同）
.venv/Scripts/python.exe -m app.core.log_query --trace 3dd63c53004e43c2

# 只看图检索节点出的问题，回看最近 3 天
.venv/Scripts/python.exe -m app.core.log_query --node node_query_kg --level ERROR --days 3

# 按关键词搜消息
.venv/Scripts/python.exe -m app.core.log_query --grep 触发断崖 --days 7 --limit 50

# 原样输出 JSON，喂给别的工具
.venv/Scripts/python.exe -m app.core.log_query --trace 3dd63c53004e43c2 --json
```

查询入口刻意做成了**命令行模块而不是只写 jq 用法**：本机没装 jq，只在文档里贴 jq 命令
等于给了一个跑不通的示例。装过 jq 的话，等价的写法是
`jq -c 'select(.trace_id=="...")' logs/*.jsonl`。

**字段固定**：`ts` / `level` / `module` / `function` / `line` / `message`
\+ 归因四件套 `trace_id` / `tenant_id` / `session_id` / `node`
\+ 业务自己 bind 的 `extra` ＋（有异常时）`exception{type,message,traceback}`。
归因字段**一定存在**，不在请求里时是空串而不是 `null`——这样 `select(.trace_id != "")` 这类查询不用额外判空。

**消息开头重复的 `[节点名]` / `[函数名]` 会被自动去掉**
节点代码沿用了 `logger.info(f"[{NODE_NAME}] [{function_name}] ...")` 的写法，而在**入口函数**里
这两个名字恰好相同——消息自己就重复了一遍，再加上行首标签与 `module:function`，
一整行里节点名能出现四次。实测**图内 41% 的日志如此**。现在补丁会剥掉开头连续的
`[节点名]` / `[函数名]`：

```
改前：INFO | [59a0349a·node_item_name_confirm] node_item_name_confirm.py:node_item_name_confirm:382 - [node_item_name_confirm] [node_item_name_confirm] 开始处理
改后：INFO | [59a0349a·node_item_name_confirm] node_item_name_confirm.py:node_item_name_confirm:382 - 开始处理
```

**为什么在补丁里收口、而不是去改 230 处 f-string**：一处生效、新节点自动受益，
节点代码也不必为了日志好看而扭曲写法。只剥**开头连续**的、且内容确实等于当前节点名或
函数名的方括号——正文中间的 `[重要]`、`[图片]` 一律不动。实测改前 41% → 改后 0%。

**两种归因来自两个地方**，缺一个就只剩半个标签：`node` 由 `add_tracked_node` 在注册节点时包上，
**只要走图就有**；`trace_id` / `tenant_id` / `session_id` 由 `usage_context` 在**入口**设置。
所以五个入口（2 个服务 + 3 个命令行跑图的地方）都包了它——
命令行入口漏包过一次，症状是日志里 `[node_rerank]` 有节点名却没 trace，账本里那几笔也归不到哪一次运行。
**新增入口时照做。**

### 三个设计取舍

**归因字段复用记账那套 `request_context`，不另造 ID**
`trace_id` 在[「调用记账」](#调用记账)里已经生成，日志直接读同一个 ContextVar。
好处是**日志与账目天然对得上**：`jq` 出一个 trace 的日志、`summarize_trace()` 出同一条 trace 的账，
两边拼起来就是「这次问答每一步说了什么、花了多少」。

**注入靠补丁，不靠调用点自觉**
`logger.py` 里一个 `enrich_record` 补丁统一附着字段，**16 个节点、几十处 `logger.info` 一行都没改**。
让调用方自己带上下文是行不通的——漏了不报错，只是那几行日志悄悄少了两列。
补丁顺带把「定位真实调用位置」的栈遍历合并进来，两个效果只走一次栈。

**没有把文本日志换成 JSONL，而是并存**
出问题时人肉读 JSON 很难受，机器读彩色文本同样难受。代价是多写一个文件——
本项目日志量很小（一次问答百来行），这点 I/O 换两种读者都舒服，值得。
`LOG_JSON_ENABLE=False` 可单独关掉。

### 两个 loguru 的坑（改 `logger.py` 前必读）

JSONL 那一份**不是**在 `format` 里拼出来的，而是在补丁里预渲染好、`format` 只写一个 `{jsonl}` 占位符。
这不是绕远路，是因为 loguru 会把**可调用 format 的返回值再当模板解析一遍**，内容一旦进了模板就连踩两坑：

1. JSON 的花括号被当字段名 → `KeyError: '"ts"'`
2. 正文里的 `<frozen runpy>`、`<class 'ValueError'>` 被当**颜色标记** →
   `ValueError: Tag "<module>" does not correspond to any known color directive`

把内容留在**值**里就没这些问题：`format_map` 只把值当字符串替换，不再解析。
模板恒定还让 loguru 的格式缓存（`lru_cache(maxsize=64)`）一直命中。
`_json_format` 的注释里记了这两条的现场，别「优化」成直接返回 JSON 字符串。

另外两点：可调用 format **不会自动补换行**（字符串 format 会），换行要自己加；
`serialize=True` 虽是 loguru 内置的结构化输出，但它**丢弃所有自定义 record 字段**
（实测 `trace_id` 全变 `None`），所以用不了。

---

## 端口一览

| 端口 | 服务 |
|---|---|
| 7474 / 7687 | Neo4j Browser / Bolt |
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
| `llm_usage` | 调用账本，每次模型调用一条（append-only，`ts` / `trace_id` / `tenant_id` 索引），见[「调用记账」](#调用记账) |
| `users` | 访问密钥（只存 sha256）与租户标识，见[「访问鉴权」](#访问鉴权) |

**Neo4j**（知识图谱，按 `file_title` 隔离）

| 元素 | 说明 |
|---|---|
| `(:Entity {name, type, item_name, file_title})` | 抽取出的实体，type ∈ 部件 / 操作 / 故障 / 参数 / 其他 |
| `(:Chunk {chunk_id, title, parent_title, item_name, file_title})` | 切片节点，`chunk_id` 对应 Milvus 主键 |
| `(:Entity)-[:APPEARS_IN]->(:Chunk)` | 实体挂载到来源切片 |
| `(:Entity)-[:REL {type, file_title}]->(:Entity)` | 实体间关系，type ∈ 组成 / 导致 / 解决 / 使用 / 连接 / 参数属于 / 其他 |

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

**MCP 联网搜索：沿用教程的异步 SDK，但传输类必须换、`mcp` 必须钉在 1.x**
整体按教程写（`openai-agents` 的 MCP 客户端 + `asyncio` 桥接），但有三处不得不偏离：

- **传输类换成 `MCPServerStreamableHttp`**：教程的 `MCPServerSse` 连 `/sse`，而本服务的 `/sse` 返回 200 后**一个字节都不推**（实测挂起 25 秒无输出），该客户端依赖服务端先发 `endpoint` 事件，根本用不了。改连 `/mcp`。
- **`mcp` 必须 `<2`**：mcp 2.x 改用 `server/discover` 新握手（协议 `2026-07-28`），百炼服务端仍是 `2024-11-05` 老协议，收到直接回 **HTTP 500**。**升级 mcp 会静默打断联网搜索**，`pyproject.toml` 已加约束。
- **工具名与参数**：本服务只有 `search_pro`，且**只接受 `query`**；照教程传 `count` 会直接 `isError`。

还有一处教程没覆盖的坑：非流式路径下 `run_query_graph` 是在 `async def` 路由里**直接被调用**的，此时事件循环已在运行，`asyncio.run()` 会抛 `RuntimeError`。`mcp_search_utils._run_coro` 检测到这种情况就另开线程执行（流式路径走 `BackgroundTasks` 线程池，不受影响）。

**RRF 只融合切片类召回，联网结果留给重排**
联网搜索返回的是 `{title, url, snippet}`，**没有 `chunk_id`**，而 RRF 靠 `chunk_id` 跨路去重计分，硬塞进去只会被当成无效项丢弃。教程的设计正是如此分工：RRF 管同源融合（基线 / HyDE / 图谱，都是 Milvus 切片），跨源合并（切片 + 网页结果）交给 `node_rerank`。所以 `node_web_search_mcp` 的结果不会白做，它在重排阶段并入。

**重排走 API，动态 Top-K 的阈值按 API 的分数尺度改过**
教程用本地 BGE（`FlagEmbedding` 的 `FlagReranker`），本项目改用 DashScope `gte-rerank-v2`，取舍同嵌入模型——项目已是全 DashScope 架构（LLM / 嵌入 / 重排共用一个 key），本地路线要装 torch（约 2GB）+ 下载 1.3GB 模型，且本机无 CUDA 只能 CPU 推理，而重排每次查询都要跑。

**两者的分数尺度不同，教程的阈值不能直接搬**：

- 本地 BGE 返回**无界 logits**，教程的断崖阈值是按这个尺度调的
- `gte-rerank-v2` 返回 **0~1 归一化分数**，实测真实查询下相邻最大落差仅约 0.14

据此在 `node_rerank.py` 顶部改了两处常量：

- **去掉绝对阈值 `GAP_ABS=0.5`** —— 在这个尺度下永远不会触发，是死参数，只保留相对阈值 `GAP_RATIO=0.25`
- **`MIN_TOPK` 由 1 抬到 3** —— 截断循环从 `MIN_TOPK-1` 起探测，`MIN_TOPK` 因此是硬地板；教程的 1 曾导致 5 条候选因 0.79 → 0.38 的陡降被截到只剩 1 条，答案生成靠一条切片支撑显然不够

调参时要先用真实查询统计分数分布，不要凭感觉改。

**图谱可视化首次引入外部 CDN，并用三重措施兜住**
项目前端此前是**零外部依赖**（所有脚本样式内联在单个 HTML 里，无静态目录）。图谱渲染需要图库，权衡后选了 ECharts（力导向布局效果好、中文文档全、代码量最小），代价是打破了这个惯例。

三条措施降低风险：

- **三个 CDN 源按序回退**，且都是在本机实测过可完整下载的（npmmirror 0.66s → bootcdn 8.2s → jsdelivr 13.6s；`staticfile` 实测连接被断故未收录）
- **懒加载**：只在首次点「图谱」时才拉取，`Promise` 缓存避免重复请求
- **文本降级**：三个源都失败时（本机环境有 Clash 拦 CDN 的先例）不白屏、不抛未捕获异常，改为展示类型分布 + 关联最多的 Top20 实体，信息照样能看

另外两点渲染取舍：**切片不作为节点画进图**（84 个切片会把 257 个实体淹没），改为悬停实体时用 tooltip 告知它出现在几个切片里；**标签默认不显示**，只在悬停与缩放 ≥1.5 倍时出现，否则 257 个中文标签必然糊成一团。

**答案的 Markdown 在前端手写渲染，不引库**
模型会输出 `**粗体**`、有序/无序列表、嵌套子步骤、`>` 引用块。原先前端用 `textContent` 直接显示，这些标记原样露出、观感很差。

没有引 marked.js 之类的库，而是写了个约 100 行的渲染器（`chat.html` 的 `renderMarkdown`）。覆盖：标题、代码块、粗体/斜体/行内代码、链接、有序与无序列表（含一层嵌套）、引用块、分隔线、段落——比模型眼下实际用的多一些，留了余量。

两个安全点：

- **先把整段 HTML 转义再解析**。答案里夹着文档正文，不转义等于把切片内容当标签执行（已实测 `<img onerror>` 与 `<script>` 都被转义成文本、不触发）
- **链接只放行 `http(s)://` 与本域路径**，挡掉 `javascript:` 之类的伪协议；外链带 `rel="noopener"`

流式过程中**每收到一块就整段重渲染**，而不是追加文本节点：Markdown 上下文相关，列表要凑齐才能成块，追加渲染会先冒出一堆裸标记。实测流式期间加粗就已生效，不会等 final 才「跳变」。

**答案的配图单列成图卡，且只放行参考内容里真实出现过的链接**
`prompts/answer_out.prompt` 要求模型在答案末尾追加一个【图片】区块。`node_answer_output` 把它拆出来单独作为 `images` 返回、并从正文里去掉——用户不必看到一堆裸链接。

每个元素是 `{url, caption}`。**caption 优先取该图的 alt 文本**——导入时 `node_md_img` 已经让多模态模型逐张看过图，把描述写进了 Markdown 的 alt 里（如「打开烫金膜盒支架盖，按箭头方向将烫金膜盒插入支架」）。alt 为空或是占位（`图片`）时才退回用章节标题。

**为什么不直接用章节标题**：同一个章节下常有多张不同的图，用标题会让它们图注一模一样——实测 5 张图的图注只有 2 种，其中一种还是「向下轻推烫金膜盒，直到它锁定到位。」这种对图本身毫无说明的句子。改取 alt 后，5 张图得到 5 条各自准确、互不相同的描述。

前端据此渲染成「图 N + 名称」的图卡（带边框、点击看原图），并用一条分隔线与正文隔开，避免几张裸图突然堆在末尾。

**为什么要白名单过滤**：模型可能编造或改写链接，直接透传会让前端显示一排破图。上面那张「url → 图注」映射正好充当白名单，不在其中的一律丢弃并记 warning——**一个结构同时解决图注与过滤两件事**。

流式模式下**图片区块不推给前端**：一旦读到 `【图片】` 标记就停止推送 delta（但仍继续累积原文，否则解析不出链接），所以打字机效果不会闪过一段裸 URL。

**标记可能被切在两个 chunk 之间**（先到「【」、下一块才是「图片】」），所以推之前会先扣住
`len("【图片】")-1` 个字符不推——它们随时可能是标记的前缀。这条曾经漏掉：那个孤零零的
「【」会跟着打字机闪过去，直到 final 覆盖才消失。修完后不变量是
**「推出去的正文 == 最终答案里图片区块之前的部分」**，四个边界场景（切块 / 无标记 / 整块 /
流停在半个标记上）都在 `node_answer_output._check_stream_boundary()` 里离线跑，不调接口。

**一片都没检索到时直接兜底，不调 LLM**
`reranked_docs` 为空时返回固定的「没有找到相关内容」而不是让模型自由发挥——没有参考内容时它只会编。

**图谱检索只回 chunk_id，正文回 Milvus 取**
图谱的 `:Chunk` 节点只存 chunk_id 与标题、**不存正文**——正文留在 Milvus。这样图谱不必重复存一份文本，代价是查询时多一次 Milvus 批量取（用的正是 `fetch_chunks_by_chunk_ids`，它本来就是为「只有 chunk_id 没有文本」的场景准备的）。

查询策略是「**种子 + 一跳邻居**」：问题里直接出现的实体名作种子（权重 2），种子的一跳邻居作扩展（权重 1），再按实体 df 打折。一跳扩展正是图相对纯向量检索的价值——能捞到语义上并不相似、但通过关系关联的切片。

值得留意的是，打分里的 df 取自图谱，反映的是**抽取覆盖度**而非真实词频，所以它作为「特异性」信号并不可靠。试过 `1/√df` 与 `1/df`，都无法让排序完全精确。但在当前架构下这可以接受：**RRF 只按排名融合**（rank 1 与 rank 2 的贡献差不到千分之一），真正的精度由下游重排按正文语义决定，图谱这一路的价值在于**把正确切片捞进候选集**——实测确实做到了。

**知识图谱按文档隔离，不做跨文档实体合并**
每个 `Entity` / `Chunk` 都带 `file_title`，同一实体出现在两篇文档里就是两个节点。这牺牲了跨文档的实体归并，换来的是**清理简单且安全**——一条 `MATCH (n) WHERE n.file_title = $ft DETACH DELETE n` 就够，不会误删其他文档。这和 Milvus 幂等清理只按 `file_title` 是同一条原则。

代价是硬约束：**不存在跨文档共享节点**。将来若真要做实体合并，这套清理会立刻失效，届时要改成按文档记录拥有关系再删。

另外两点设计：

- **关系类型是 `:REL` 上的属性、不是关系标签**。Cypher 无法参数化关系标签，拼字符串既有注入风险又无法约束取值，所以统一用单标签 + `type` 属性，取值在 Python 侧按白名单兜底成「其他」
- **不建 `:Product` 节点**。每篇文档只有一个 `item_name`，做成实体属性即可；「按产品找文档」已由 Milvus `kb_item_names` 承担，图谱不重复存一份

**图谱抽取必须「先抽取、再清理、最后写入」**
Milvus 每次重新入库都会生成**全新的 chunk_id**，重复导入时旧 `:Chunk` 节点带的是失效 id、`APPEARS_IN` 会变成指向幽灵切片的悬空边，所以写前必须清理。但顺序不能颠倒：先清后抽的话，一次抽取失败就把已有图谱清空了，比残留更糟。

还有个坑值得记：LLM 会在 `relations` 里引用没登记进 `entities` 的名称（例如把「装入烫金膜盒」当 src 却不在实体列表里），而 Cypher 的 `MATCH` 找不到端点会**静默丢边**。除了在提示词里明确要求「被关系引用的名称必须先在 entities 里登记」，代码侧还会把缺失端点补登记为实体（type 记「其他」），并统计补的数量——补得过多说明提示词在漂移。

---

## 已知问题 / 待办

| 项 | 说明 |
|---|---|
| 图谱侧排序只是近似 | `node_query_kg` 按「种子实体权重 2 / 一跳邻居 1，再除以 df」打分，但 **df 反映的是抽取覆盖度而非真实词频**（`HAK 180` 只被抽到 2 个切片里），所以排序不精确。好在 RRF 只看排名、精度由下游重排决定——图内实测正确的切片都进了候选集 |
| 答案配图未做去重与尺寸处理 | 同一张图可能因多切片命中而重复出现在 `images` 里；大图直接原样加载，未生成缩略图 |
| 自带图测试场景1恒失败 | `main_graph.py` 的 `__main__` 拿「烫金膜盒怎么安装？」（不带型号）当查询，产品名确认必然判拒识，四路检索全被跳过。这是既有缺陷，与该测试想验证的图拓扑无关；换成完整产品名即可通过 |
| `get_recent_messages` 曾取错数据 | 原实现 `sort(ASCENDING).limit(N)` 取的是**最旧** N 条，已修为倒序取再反转为正序 |
| 相似度阈值 | `kb_item_names` 的 0.85/0.6 阈值取自教程代码（教程正文写的是 0.95，两处不一致） |
| 无引用的模块 | `format_utils.py`、`mongo_history_utils_new.py` 均无引用 |
| 重排相对阈值验证样本少 | `GAP_RATIO=0.25` 由教程继承（相对值可跨尺度迁移），但只在少数真实查询上验证过，候选规模变化后可能仍需微调 |
| 计价表是快照 | `pricing_config.py` 里的单价取自百炼 2026-10 的价目表；阶梯计价只按最低档算。tokens 是原始事实，单价更新后报表会自动按新价重算，但**表本身要人工跟** |
| 账本有两处不计成本 | 联网搜索按次计费、单价未公开（账本记次数、成本标为「未计价」）；MinerU 按页数配额计费、与 token 无关，不在账本内 |
| 导入链路的记账未做端到端验证 | 归因机制与检索链路完全相同（已实测 8 次调用全部正确归到节点），但没跑整篇文档导入去验证——那要真调 MinerU、耗时数分钟、消耗解析配额 |
