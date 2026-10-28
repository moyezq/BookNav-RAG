# 目标架构

核心功能、核心模块、暂时或明确不做。

## 1. 系统是什么

把约定格式的文档变成一份知识库，按印刷目录把每页归到对应章节、抽正文并自动抽出过滤标签，再按检索作答。输入是查询文本，输出是答案文本与页码引用。默认检索单元是知识块。不是 SaaS 管理平台，不从产品仓 fork。样本可以是汽车手册；手册不是库的领域定义。

一份逻辑知识库，可含多个来源。来源是文档身份，也是唯一强制筛选维。过滤标签是入库从正文抽出的 `{field, value}`，字段名不预置；查询时按命中条数给候选加权，不强制筛选。KV 条目是可拆支线：存储可由实验脚本写入，查询侧不进召回。

源格式目标态仍含 PDF、Office、纯文本 / Markdown、URL / HTML；本阶段路由只登记 PDF。音视频明确不做。

在线主入口是一个轻量 HTTP 接口。命令行调试走同一条图。检索门面是独立前端 `frontend/`，只调该接口展示问答。编排用 LangGraph 把各层依次调用；生成节点用 LangChain `prompt | model | parser`。`create_agent` / `create_deep_agent` 不进主路径。

**明确不做：**

- 主路径 Agent / Deep Agent / GraphRAG：RAG 是检索后生成，不是模型选工具。
- RBAC、任务队列、管理 UI、产品级 HTTP API 网关：一个轻量 POST + 一页对话。
- 把样本领域写成约定字段（`brand` / `car` 列）。
- 评测台产品：只留最小评测口。
- 自建解析微服务；把 anyurl2text / MinerU / anydoc 源码引进本仓。
- 按格式堆源适配器。
- 为完整而双写 KV；把 KV 再向量化当默认。
- 音视频入库。
- 在 `extract/` 或 `vector/` 写入库总编排；把向量模型服务实现放进 `offline/` 或 `online/`。
- 为向量匹配再开关系库或文档库。

状态：**已实现** = 代码位置与目标相符；**内联** = 逻辑有但未拆开；**未做** = 目标有、代码没有；**暂时不做** = 目标留预留位置，本阶段不排期；**明确不做** = 只标边界。

## 2. 离线 · 入库

人对库入口是 `scripts/ingest/ingest.py`：按份先抽取再向量。两层不互相 import，接点是磁盘上的 `chunkable_source.json`。

```
offline/extract/            抽取层：一份源 → 可分块原文。内部 probe → route → 方式 → enrich → collect。  已实现
├── probe.py                源路径 → 基础信息。  已实现
├── route.py                仅 PDF 放行；有印刷目录则方式名 handbook，否则 pdf。  已实现
├── methods/pdf.py          pymupdf4llm 抽页；印刷目录动态扫描。  已实现
├── methods/anyurl.py       预留实现文件，路由不登记。  暂时不做
├── methods/vlm_ocr.py      预留实现文件，本阶段不调用。  暂时不做
├── enrich.py               从正文抽动态 tags。  已实现
├── collect.py              写出 chunkable_source.json。  已实现
└── entrance.py             串上面五步。  已实现

offline/vector/             向量层：可分块原文 → LanceDB chunks 表。HTTP 调 8030。  已实现
├── chunk.py                有目录则按标题拼块；页码块级。  已实现
├── store.py                按来源写入；Hash / HTTP 客户端同文件。  已实现
└── entrance.py             读可分块原文，切批 embedding，写库成功后才写 step4。  已实现

offline/store/kv_store.py   KV 支线存储。查询侧不接。  暂时不做
```

## 3. 在线 · 问答

主入口 `POST /api/ask`。检索门面 `frontend/` 只调该接口；没有该页，接口与命令行仍能问。过滤是层，图上不是节点。

```
online/
├── server.py               问答 HTTP：创建应用、注册路由、lifespan load 一次库。  已实现
├── router.py / schema.py   轻量 HTTP。  已实现
├── graph.py                LangGraph：intent →（闲聊则跳过）recall → rerank → generate。  已实现
├── intent/                 分类、改写 / 跟进补全、标签候选。历史只进本层。  已实现
├── filter/                 来源强制筛选；标签按命中条数加权。  已实现
├── recall/                 装库、混合检索、词面兜底。不拆文件。  已实现
├── rerank/                 截 TOP_K（不换序），再并同节邻块。  已实现
├── prompt/                 命中 → 提示变量与 citations。  已实现
└── generate/               闲聊 / 拒答 / 按命中生成。  已实现

frontend/                   检索门面：独立前端，只调 8031；视觉规范 DESIGN.md。  已实现
scripts/ask/query_kb.py     单轮调试。  已实现
scripts/ask/chat_kb.py      多轮同一 session。  已实现
scripts/eval/eval_kb.py     最小评测口，走 ask_once_core，停在生成前。  已实现
```

## 4. 共享 · 向量模型服务

入库和提问都 HTTP 调 8030。实现不属于 `offline/` 也不属于 `online/`。

```
embed/server.py             向量 HTTP 入口与实现。  已实现
embed/download_embed.py     权重下载到同目录 models/。  已实现
embed/models/               本地 embedding 权重（不进仓）。  已实现
```

## 5. 仓库根

```
start.py                    一键起 8030 / 8031 / 8032。  已实现
pyproject.toml              包清单；uv 管理环境。  已实现
conf/base.py                ROOT、层开关与阈值；在线布尔只剩 USE_REWRITE。  已实现
conf/config.yaml            相对 data / lancedb 目录名。  已实现
scripts/ingest/ingest.py    人对库入库；VECTOR_ONLY 只重跑向量层。  已实现
scripts/ingest/verify_ingest.py  只读验收；硬断言未过则非零退出。  已实现
tests/                      纯函数与替身，不调 LLM、不调 8030。  已实现
```

**暂时不做**（目标留预留位置，不进当前排期）：

- 图片 OCR：`vlm_ocr.py` 在，流程不调用。
- 入库后 ask 进程 reload：启动仍 load 一次。
- 交叉编码器重排：重排节点只截断，不换序。
- Agent 学习沙箱：永不接提问口。
- 拆共享 Chat 客户端：`chat_text` 仍住在生成包。
