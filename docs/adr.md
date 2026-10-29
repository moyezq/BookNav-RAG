# 架构决策（现行）

仍有效的为什么。

## ADR-0002 · 2026-10-29 · 知识块是唯一检索单元

仓里曾有手册进知识块、URL 抽 KV 两条互不相通的离线流程。选定知识块走向量 + BM25。短事实也在同一批正文里检索，不另建精确匹配库。KV 支线已删，见 ADR-0040。

## ADR-0003 · 2026-10-29 · 在线主路径固定，对象形状稳定

在线问答按层依次调用，不写成一条巨函数。主路径：意图识别 → 召回 → 重排截断 → 生成。启用与删除插件用同一套数据对象。现在只有改写这一个开关能关（`USE_REWRITE`）。拒答一直开着。不要把词重叠重排接回提问口。

## ADR-0004 · 2026-10-29 · 知识块检索默认向量加 BM25

检索公式默认混合：LanceDB 余弦近邻加进程内 BM25，再 RRF。不上独立搜索引擎。向量只走 8030 的语义 embedding；服务挂了退化为纯 BM25。不另造假向量。

## ADR-0006 · 2026-10-29 · 抽取接点是可分块原文

抽取与分块之间的对象是可分块原文：正文、可选页列表、可选目录区间、来源元数据。由 collect 写出 `chunkable_source.json`。页码留下来，引用才能写第几页。pymupdf4llm 的落盘是原始页，不替代这个接点，见 ADR-0039。

## ADR-0007 · 2026-10-29 · 模块 I/O 约定在数据对象层

插件装上或删掉时，字段仍一致。约定的对象为：可分块原文、知识块、查询、检索命中、生成结果。不在目标架构里写 HTTP 或类定义。

## ADR-0008 · 2026-10-17 · 一份知识库，来源是文档身份

逻辑上只有一份知识库；文档身份是来源，检索默认全库。来源可强制筛选。过滤标签不是库边界，也不用来强制筛选（加权见 ADR-0035）。字段名不预置，见 ADR-0026。

## ADR-0009 · 2026-09-28 · 本仓是 RAG 问答系统

把约定格式的文档送进一份知识库，再按检索作答。输入查询文本，输出答案与引用。对照附录只作研究材料。不 fork 产品仓。不做管理台、任务队列、登录鉴权。

## ADR-0010 · 2026-10-29 · 在线主路径用 LangGraph

在线图是确定性节点链。对话模型 HTTP 只在 `service/deepseek`，生成层不再自持客户端。`create_agent` / `create_deep_agent` 只进 `scripts/` 的学习预留位置，不接进提问口。检索不是 Tool。

## ADR-0011 · 2026-10-28 · 代码按离线层与在线层分目录

新代码进各层的预留位置：抽取 `backend/offline/extract/`，向量 `backend/offline/vector/`，在线各层在 `backend/online/`，向量模型服务在 `service/embed/`。仓库根按 backend / frontend / docs / service 分地盘，见 ADR-0038。不套 `src/`，不用产品仓目录名。不重建 `query/`。

## ADR-0012 · 2026-09-28 · 在线主入口是 HTTP，引用在侧栏

一个 POST 接口的提问口为主入口；同一会话多轮；历史只给意图识别；侧栏展示当前选中那条助手消息的引用。命令行只留作调试。不做管理台，不升级成开放 API 网关。

## ADR-0014 · 2026-10-17 · 过滤标签入库抽出，在线由意图给候选

入库时从知识正文抽出过滤标签；来源仍是文档身份。在线由意图识别对照库内已有标签给出本轮候选，供召回加权，不手填、不按字段强制筛选（ADR-0035）。问答页只展示芯片。命令行 `filters` 非空时可覆盖，方便调试。字段名见 ADR-0026。

## ADR-0015 · 2026-10-28 · 提问口不另开 query/

HTTP 路由在 `backend/online/router.py`；进程入口 `python -m backend.online.server`。命令行调试 `scripts/ask/query_kb.py`，评测 `scripts/eval/eval_kb.py`。不重建 `query/`。

## ADR-0016 · 2026-10-29 · 抽取层四个 class，collect 写出可分块原文

抽取层按数据处理场景拆 class：`PdfPageExtract` 抽页、`CatalogueBuild` 建印刷目录区间、`TagEnrich` 抽过滤标签、`ChunkableCollect` 写出可分块原文。不为 Office 预留空文件或空登记表；以后加格式就加对应 class 和脚本步骤。anyurl2text 源码不引进。`handbook` / `pdf` 是可分块原文上的方式名（有没有目录区间），不是路由插件名。测试与调试在 `scripts/`。

## ADR-0017 · 2026-10-29 · 分块、向量化、向量存储合并成向量层

离线后半段是一层：可分块原文进，向量库出。目录 `backend/offline/vector/`。与抽取层用磁盘上的 `chunkable_source.json` 相接。权威库是 LanceDB，见 ADR-0024。

## ADR-0018 · 2026-10-28 · 向量模型服务独立于离线目录

入库和提问都要把文本变成向量。实现与进程入口都在 `service/embed/server.py`，不属于 `backend/offline/` 也不属于 `backend/online/`。从仓库根 `python -m service.embed.server`。下载是 `service/embed/download_embed.py`，权重落在同目录 `service/embed/models/`（不再套一层模型名目录）。离线与在线都 HTTP 调 8030。

## ADR-0019 · 2026-10-29 · 人对库两个脚本：extract_pdf.py 与 ingest.py

人对库先跑 `python scripts/ingest/extract_pdf.py`（pymupdf4llm → 原始页），再跑 `python scripts/ingest/ingest.py`（原始页 → 可分块原文 → 向量库）。抽取层提供 class，向量层提供函数；脚本按步骤函数编排，不把总编排写进 `extract/` 或 `vector/`。要跑哪一份源，改该脚本顶部常量，不是系统配置。验收仍是 `verify_ingest.py`，演示 PDF 仍是 `make_demo_pdf.py`。

## ADR-0020 · 2026-10-28 · 问答入口是 backend/online/server.py，路由进 backend/online/router.py

`backend/online/server.py` 只创建应用、注册路由、lifespan。`POST /api/ask` 与 schema 在 `backend/online/router.py`、`backend/online/schema.py`。该进程唯一 `__main__` 在 `server.py`；不要 `python backend/online/router.py`。

## ADR-0021 · 2026-10-29 · 密钥与第三方地址进 .env

密钥模板 `.env_example`，本仓 `.env` 不进 git。第三方 URL 与密钥只走仓库根 `.env`。不写进 `backend/conf/config.yaml`。只读本仓 `.env`：`backend.conf.apply_env()` 是唯一读取点，顺序是「进程环境变量 → 本仓 `.env`」，只填空缺。抽取解耦（ADR-0034）后已删除按 `ANYURL2TEXT_ENV` 读外仓 `.env` 的兜底；外仓地址也不再进本仓 `.env`。

## ADR-0022 · 2026-10-28 · ROOT 在 base.py，host/port 在启动文件

从本机推算的 ROOT（backend 目录）与 REPO_ROOT（仓库根，写入 `sys.path`）只在 `backend/conf/base.py`。本服务 host/port 写在 `backend/online/server.py` / `service/embed/server.py`。`config.yaml` 只放相对数据目录、向量库目录。权重目录不进 yaml，是 `service/embed/models/`（与 `service/embed/download_embed.py` 同目录）。第三方与密钥进仓库根 `.env`。层与模块常量见 ADR-0029。

## ADR-0023 · 2026-10-17 · 检索门面指独立前端

查库是召回层。检索门面指独立前端 `frontend/`，只调后端的轻量 HTTP 接口展示问答。没有该页，`POST /api/ask` 与 `scripts/ask/query_kb.py` 仍能问。不要把 `retrieve.py` 叫门面。

## ADR-0024 · 2026-09-28 · 向量库后端用嵌入式 LanceDB

Windows 本地向量写进 `backend/data/lancedb/`，一张 `chunks` 表同时存向量和命中字段。近邻走 `distance_type("cosine")`。BM25 仍进程内，load 时不把 `vector` 读进 Python。不上第二库。表列 `tags` 为 JSON，见 ADR-0026；`catalogue_title` 给同节邻块，见 ADR-0028。旧库缺这些列须删后重入库。

## ADR-0025 · 2026-09-28 · 拒答阈值用向量余弦

空命中仍拒。另有 `REFUSE_MIN_SCORE` 时看命中上的 `vector_score`（余弦）。RRF 是秩融合分，不能当相关分。纯 BM25 命中没有余弦，不当弱证据拒答。

## ADR-0026 · 2026-10-17 · 过滤标签是动态键值

来源仍是文档身份。过滤标签是入库时从正文抽出的 `{field, value}` 列表，字段名不预置。不要把样本领域（品牌 / 车型）写成表列或系统定义。标签如何参与检索见 ADR-0035（加权，不强制筛选）。

## ADR-0027 · 2026-09-28 · 生成窗口 TOP_K 由重排层截断

两路检索各取候选窗口（默认 20），融合也截到该窗口；`rerank_hits` 再截 `TOP_K`。重排只截断、不换序，换交叉编码器时不必改召回截断位置。

现状：文件仍叫 `backend/online/rerank/overlap.py`，函数不换序。

## ADR-0028 · 2026-09-28 · 重排后按目录标题并同节邻块

命中一小节后，生成需要同一目录标题下的前后步骤。`catalogue_title` 写入 LanceDB 表，不进检索命中的字段约定。重排截 `TOP_K` 之后，按来源 + 标题把邻块并进命中 `text` / `pages`，有 token 上限；空标题不并。同一标题在窗口里只留最先出现的那条。

## ADR-0029 · 2026-10-28 · 层与模块常量集中在 backend/conf/base.py

ROOT 与层、模块的开关、阈值、提示词都进 `backend/conf/base.py`，按层注释分隔。相对 data 目录仍走 yaml；权重目录是 `service/embed/models/`，不进 yaml；第三方与密钥仍走 `.env`；host/port 仍写在启动文件。不要写回各层顶部常量。入库样本开关留在 `scripts/ingest/` 各脚本顶部。

## ADR-0030 · 2026-10-29 · 删掉不起作用的开关，不修通假路径

布尔若拨了不改路径，或开了仍静默走原路，就删掉开关和对应死代码。提问口不接词重叠重排。拒答无总闸，空命中或低余弦用拒答文案。在线布尔只留真能改路的 `USE_REWRITE`。抽取层无开关。

## ADR-0031 · 2026-10-29 · 抽取不按文件名切换

不按样本名单在两种抽取之间切换。印刷目录 PDF 方式名记 `handbook`；无目录记 `pdf`。`anyurl` 是已退役的方式名。汽车手册是有目录 PDF 的样本，不是抽取层的开关。

## ADR-0033 · 2026-10-17 · 现行为什么写在 adr.md，废决策写在 history.md

分文件 ADR 加现状补丁会让旧选定继续束缚开发，历史也不完整。选定：仍有效的为什么写在一份带编号与日期的 `docs/adr.md`；被取代或废弃的改动按时间记入 `docs/history.md`。写代码听目标架构和 `AGENTS.md`，不要把 history 里的旧选定当成不许做的事。

## ADR-0034 · 2026-10-29 · PDF 本仓抽页；手册抽取与 anyurl2text 解耦

入库要对 PDF 独立抽页并写入向量库，不能依赖外仓是否已经启动；外仓 `/rag/extract/file` 已下线。选定：PDF 走 `PdfPageExtract`（pymupdf4llm）写成原始页；目录与标签在 ingest 里用 `CatalogueBuild` / `TagEnrich`。非 PDF 本阶段由 `list_source_files` 跳过。外仓源码不引进。可分块原文的约定不变。见 ADR-0016、ADR-0031、ADR-0039。

## ADR-0035 · 2026-10-10 · 过滤标签是加权信号，不是强制筛选

入库抽出的标签是文档级的、复制到该文档每个块上，只能表达「这份文档讲什么」，不能表达「这个块讲什么」。拿它强制筛选，两头都坏：同字段多值走 AND 时一个块只可能带其中一个值 → 命中 0 → 空命中直接拒答；单文档库上它又等价于零收窄。选定：`source` 仍强制筛选；`filters.tags` 只进排序——命中 N 条的块加 `min(N × TAG_BOOST_PER_HIT, TAG_BOOST_MAX)` 分，不命中的块不丢。命中判定分两路：问句字面出现标签值由代码判（同字段多值都收），字面都没命中才问模型补同义。取代 ADR-0026 的「强制筛选、多对 AND」那半；动态键值、字段名不预置仍有效。

## ADR-0036 · 2026-10-28 · 长驻服务入口在对应层包的 server.py

仓库根只留 `start.py` 一键编排。向量入口 `python -m service.embed.server`，问答入口 `python -m backend.online.server`。host/port 写在这两个文件。不要 `python service/embed/server.py` 或 `python backend/online/server.py`（sys.path 会指到子目录）。其它模块不留第二份 `__main__` 启动同一进程。跑完即退出的入库 / 评测 / 下载仍在 `scripts/`。取代 ADR-0013。

## ADR-0037 · 2026-10-17 · 包清单只在 pyproject.toml；环境用 uv

依赖写在 `pyproject.toml`。安装与启动走 `uv sync` / `uv run`。

## ADR-0038 · 2026-10-29 · 仓库按 backend / frontend / docs / service 分地盘

前后端同仓后，仓库根不再直接摊架构层。选定：离线抽取与在线问答进 `backend/`（内部仍按 offline / online 分层，ADR-0011；`conf/` 与 `data/` 也只给后端用）；检索门面仍是 `frontend/`；文档仍是 `docs/`；独立进程服务进 `service/`，按服务分子目录——`embed/` 是主路径向量服务（ADR-0018，不读 `backend/conf`），`kev08/` 是预备的决策模型服务（自带 pyproject，torch 版本与本仓向量服务无交集）。本阶段不接提问口、不进 `start.py`；后续计划是接进意图识别与拒答，接线时仍不把 torch 装进本仓环境。`scripts/`、`tests/`、`start.py`、`.env` 留仓库根。不把 embed 塞进 backend。

## ADR-0039 · 2026-10-29 · pymupdf4llm 产出落盘为原始页

`pymupdf4llm.to_markdown` 是离线链路里最慢的一步，和目录解析、抽标签、写库绑在一次运行里会让改分块或重抽标签也重跑抽页。选定：`extract_pdf.py` 把去掉 Rect、页码从 1 起的页列表写成 `backend/data/result/<stem>/raw_pages.json`（`source`、`name`、`pages[{page,text}]`）。`ingest.py` 只读这份文件，找不到就失败，不回头调 pymupdf4llm。可分块原文仍是与向量层的接点（ADR-0006）。

## ADR-0040 · 2026-10-29 · 去掉 KV 支线

预留的 KV 存储、实验脚本，以及「规格页单行冒号句拆成精确匹配」的方案，适用面只覆盖手册里极少几行，留在仓里会让人把节标题、过滤标签、页正文当成三套检索对象。选定：删除 `backend/offline/store/`，提问口与入库都不走精确匹配；规格句留在知识块正文里检索。取代 ADR-0005。

## ADR-0041 · 2026-10-29 · 去掉 Hash 占位向量

HashEmbedder 不能给 bge-m3 库做问答兜底（维数与表内名字都对不上），也不能提高检索精度。8030 挂了时提问口已经走纯 BM25。选定：删除 HashEmbedder 与 `EMBEDDER_KIND`；入库只调 8030。旧 hash 库 load 时报错，须删 `backend/data/lancedb` 后重入。

## ADR-0042 · 2026-10-29 · 对话模型只有 service/deepseek 一个调用方

离线抽标签和在线意图、生成都要调 DeepSeek。客户端若留在生成包，抽取层就得 import 在线层。选定：唯一调用方是 `service/deepseek/http_client.py` 里的 `HttpClient`，不另开端口，不进 `start.py`，不读 `backend/conf`。返回按官方非流式 Chat Completions 分成 kind，调用方按 kind 分支。不发流式。思考关闭。被否的备选是：各模块自己发 HTTP；继续用 LangChain 只取字符串，从而丢掉 `finish_reason`。
