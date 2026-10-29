# 目标架构

核心功能、核心模块、暂时或明确不做。

## 1. 系统是什么

把约定格式的文档变成一份知识库，按印刷目录把每页归到对应章节、抽正文并自动抽出过滤标签，再按检索作答。输入是查询文本，输出是答案文本与页码引用。检索单元是知识块。不是 SaaS 管理平台，不从产品仓 fork。样本可以是汽车手册；手册不是库的领域定义。

一份逻辑知识库，可含多个来源。来源是文档身份，也是唯一强制筛选维。过滤标签是入库从正文抽出的 `{field, value}`，字段名不预置；查询时按命中条数给候选加权，不强制筛选。检索单元只有知识块。

源格式目标态仍含 PDF、Office、纯文本 / Markdown、URL / HTML；本阶段只抽 PDF。音视频明确不做。

在线主入口是一个轻量 HTTP 接口。命令行调试走同一条图。检索门面是独立前端 `frontend/`，只调该接口展示问答。编排用 LangGraph 把各层依次调用；生成节点用 LangChain `prompt | model | parser`。`create_agent` / `create_deep_agent` 不进主路径。

**明确不做：**

- 主路径 Agent / Deep Agent / GraphRAG：RAG 是检索后生成，不是模型选工具。
- RBAC、任务队列、管理 UI、产品级 HTTP API 网关：一个轻量 POST + 一页对话。
- 把样本领域写成约定字段（`brand` / `car` 列）。
- 评测台产品：只留最小评测口。
- 自建解析微服务；把 anyurl2text / MinerU / anydoc 源码引进本仓。
- 按格式堆源适配器。
- KV 条目、精确匹配库、从正文拆「字段：值」。
- 音视频入库。图片 OCR。
- 在 `extract/` 或 `vector/` 写入库总编排；把向量模型服务实现放进 `backend/offline/` 或 `backend/online/`。
- 为抽取预留空的 `methods/` 或空登记表。
- 为向量匹配再开关系库或文档库。

状态：**已实现** = 代码位置与目标相符；**内联** = 逻辑有但未拆开；**未做** = 目标有、代码没有；**暂时不做** = 目标留预留位置，本阶段不排期；**明确不做** = 只标边界。

## 2. 离线 · 入库

人对库入口是两个脚本：`extract_pdf.py` 把 PDF 抽成原始页，`ingest.py` 从原始页做成可分块原文再向量。两层不互相 import；与向量层的接点仍是磁盘上的 `chunkable_source.json`。

```
backend/offline/extract/            抽取层：一份源 → 可分块原文。四个 class，无总编排。  已实现
├── pdf_page.py             PdfPageExtract：pymupdf4llm → [{page, text}]。  已实现
├── catalogue.py            CatalogueBuild：印刷目录 → 区间。  已实现
├── enrich.py               TagEnrich：从正文抽动态 tags。  已实现
└── collect.py              ChunkableCollect：写出 chunkable_source.json。  已实现

backend/offline/vector/             向量层：可分块原文 → LanceDB chunks 表。HTTP 调 8030。  已实现
├── chunk.py                有目录则按标题拼块；页码块级。  已实现
├── store.py                按来源写入；HTTP 客户端同文件。  已实现
└── entrance.py             读可分块原文，切批 embedding，写库成功后才写 step4。  已实现
```

## 3. 在线 · 问答

主入口 `POST /api/ask`。检索门面 `frontend/` 只调该接口；没有该页，接口与命令行仍能问。过滤是层，图上不是节点。

```
backend/online/
├── server.py               问答 HTTP：创建应用、注册路由、lifespan load 一次库。  已实现
├── router.py / schema.py   轻量 HTTP。  已实现
├── graph.py                LangGraph：intent →（闲聊则跳过）recall → rerank → generate。  已实现
├── intent/                 分类、改写 / 跟进补全、标签候选。历史只进本层。  已实现
├── filter/                 来源强制筛选；标签按命中条数加权。  已实现
├── recall/                 装库、混合检索、词面兜底。不拆文件。  已实现
├── rerank/                 截 TOP_K（不换序），再并同节邻块。交叉编码器未做。  已实现
├── prompt/                 命中 → 提示变量与 citations。  已实现
└── generate/               闲聊 / 拒答 / 按命中生成。  已实现

frontend/                   检索门面：独立前端，只调 8031；视觉规范 DESIGN.md。  已实现
scripts/ask/query_kb.py     单轮调试。  已实现
scripts/ask/chat_kb.py      多轮同一 session。  已实现
scripts/eval/eval_kb.py     最小评测口，走 ask_once_core，停在生成前。  已实现
```

## 4. 独立进程服务

`service/` 按服务分子目录。实现不属于 `backend/offline/` 也不属于 `backend/online/`。

```
service/embed/                  向量模型服务：入库和提问 HTTP 调 8030。  已实现
├── server.py                   向量 HTTP 入口与实现。  已实现
├── download_embed.py           权重下载到同目录 models/。  已实现
└── models/                     本地 embedding 权重（不进仓）。  已实现

service/kev08/                  决策模型预备服务：独立 uv 环境与权重，不接提问口。  暂时不做
```

## 5. 仓库根

```
backend/                   离线抽取与在线问答。内部仍按 offline / online 分层。  已实现
backend/conf/               ROOT（backend 目录）、层开关与阈值。  已实现
backend/data/               源 PDF、抽取产物、LanceDB。  已实现
frontend/                   检索门面。  已实现
service/embed/              向量模型服务。  已实现
service/kev08/              决策模型预备服务。  暂时不做
docs/                       目标架构与 ADR。  已实现
start.py                    一键起 8030 / 8031 / 8032。  已实现
pyproject.toml              包清单；uv 管理环境。  已实现
scripts/ingest/extract_pdf.py  pymupdf4llm → raw_pages.json。  已实现
scripts/ingest/ingest.py    原始页 → 可分块原文 → 向量库。  已实现
scripts/ingest/verify_ingest.py  只读验收；硬断言未过则非零退出。  已实现
scripts/ingest/make_demo_pdf.py  自造演示手册 PDF。  已实现
tests/                      纯函数与替身，不调 LLM、不调 8030。  已实现
```

**暂时不做**（目标留预留位置，不进当前排期）：

- 入库后 ask 进程 reload：启动仍 load 一次。
- Agent 学习沙箱：永不接提问口。
- kev08 接进意图识别（闲聊 / 结束 / 查库）：目录在 `service/kev08/`，本阶段不调用、不进 `start.py`。结束是对话收尾，不是召回后的拒答。
- 拆共享 Chat 客户端：`chat_text` 仍住在生成包。
