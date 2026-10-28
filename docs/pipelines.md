# 主流程实现细节

各层改代码时必须知道的判定、上限、字段落点。层与目录见 `architecture.md`。

## 1. 离线入库

入口 `python scripts/ingest/ingest.py`。每份源：抽取层再向量层。前提：`python -m service.embed.server` 已启动。改完分块策略只重跑向量层：把该脚本顶部 `VECTOR_ONLY` 改成 `True` 再跑（重复使用已有 `chunkable_source.json`，不重跑抽取，也不重跑 enrich）。层 `entrance.py` 只提供函数，没有 `__main__`。

只认 `SOURCE_ALLOWED_SUFFIXES`（本阶段 `.pdf`）；不在清单里的文件跳过并打日志。要跑哪几份，改脚本顶部 `PROCESS_ONLY`（空元组 = `backend/data/source` 全部）。非 PDF 在路由拒绝。

```text
scripts/ingest/ingest.py
  extract.entrance.run_extract
    probe → route → methods → enrich → collect    # → chunkable_source.json
  vector.entrance.run_vector
    chunk → POST 8030 → store                     # → backend/data/lancedb chunks 表
                                                  # 写库成功之后才写 step4_chunks.json
```

### 1.1 抽取

`methods/pdf.py`：pymupdf4llm 抽页，把 0 起页码转成阅读器页码（从 1 起）。印刷目录**动态扫**：带表头的页 + 紧邻续页，连续 `EXTRACT_CATALOGUE_STOP_PAGES` 页不是目录页就停；硬上限 `EXTRACT_CATALOGUE_PREVIEW_PAGES`。响应带目录区间时方式名记 `handbook`，无目录记 `pdf`。`anyurl.py` / `vlm_ocr.py` 本阶段流程不 import。

抽取方式只吐文件内容（有页则 `pages[{page,text}]`，无页则 `text`；目录区间可选）。方式不写可分块原文、不抽过滤标签、不分块、不向量、不碰 KV。增方式：加 `methods/` 文件并在路由登记一行。删方式：去掉登记。不按样本文件名切换。

`enrich.py`：从正文抽出 `tags[{field, value}]`，最多 8 条；字段名随文档变，不预置。不把来源身份写成标签。抠 JSON 用花括号配对逐段试解析。抽空打原因与模型原始输出，入库汇总按名字单列，不许混进「成功 N」。实现上借 `backend.online.generate.answer.chat_text`。

`collect.py` 写出可分块原文：必有 `source`、`kind`，以及 `pages[{page, text}]` 或无页时的 `text`；`tags` 可 `[]`。可选：目录区间。可选不进检索约定：`extract_method`、基础信息摘要。

### 1.2 向量

有 `pages` 且有目录区间：先按目录分页再按标题拼知识块；否则按段 / 页切。一节超 `MAX_DOC_TOKENS` 切出的每个子块只声明自己覆盖的页（页码块级；任一页只被一个块声明）。整份文档的 tags 写到每个知识块上。不得用手填常量或文件名冒充。`catalogue_title` 写入表列，不进检索命中约定。

客户端按 `EMBED_REQUEST_BATCH` 切批。写库前校验向量维数：`EMBED_DIM_BY_NAME` 登记 embedder → 期望维数；首次建表也过这一关。按来源替换写入 `backend/data/lancedb` 的 `chunks` 表（含 `tags` JSON、`catalogue_title`）。不上第二库。不为每个标签字段加一列。

BM25 与向量并列，对同一批知识块建倒排；倒排在召回 load 时用同一批正文建，不上独立搜索引擎。Hash 向量只为打通流程，不是语义 embedding。换 embedding 维数或 embedder 名须删掉 `backend/data/lancedb` 后重新入库。`EMBEDDER_KIND=hash` 仍可跑通，但不是语义检索。

KV 抽取吃可分块原文，不吃知识块；实验脚本是 `scripts/url_to_knowledge.py`，不进抽取 `entrance`、不进召回。

## 2. 在线问答

长驻三个进程：向量 8030、问答 8031、Vite 8032。问答入口永远 `python -m backend.online.server`：该文件只创建 FastAPI、`include_router`、注册 lifespan；`POST /api/ask` 在 `backend/online/router.py`。不要 `python backend/online/router.py`。启动 lifespan：`open_knowledge_base` 一次；重新入库后要重启该进程。改完在线代码同样要重启，MemorySaver 会话会丢。

HTTP 入 `AskBody{text, session, filters}`，出 `{answer, citations, tags}`。`session` 限 `SESSION_ID_MAX_CHARS`、只含 `[A-Za-z0-9_-]`。命令行 `scripts/ask/query_kb.py`（单轮）与 `scripts/ask/chat_kb.py`（多轮同一 session）共用 `run_ask_graph`。

```text
intent →（route=chitchat 则跳过召回）recall → rerank → generate
```

闲聊与检索生成都用用户原文 `question`；召回用改写后的 `text`。身份追问沿用上一轮筛选和检索问句。`tags` 在 intent 写出，generate 不改。关掉 `USE_REWRITE`，主路径对象形状不变。

同一条检索核心另有 `ask_once_core`：与 intent / recall / rerank 同一顺序，停在生成前，给 `scripts/eval/eval_kb.py` 与 pytest 用。

### 2.1 意图识别

历史在 `turns` 里，只进入本层。`format_turns` 分类用全历史，改写只用检索轮；`HISTORY_MAX_TURNS` 截意图用历史，MemorySaver 里 turns 只留 `SESSION_TURNS_MAX`。

- `classify_route`：`retrieve` 或 `chitchat`。有历史且问文档身份则 retrieve。闲聊不是拒答。
- `rewrite_query`：默认开。关掉且不像跟进则原样。跟进补全只吃检索轮，且问句要像指代（「它」「那个」「呢」）。
- `infer_query_filters`：对照 `tag_catalog()` 给出 `{source, tags[{field, value}]}`。source 空串 = 不限来源；tags 空 = 不加权。问句字面命中目录值就直接返回（最短 `TAG_LEXICAL_MIN_CHARS`）；调用方 `filters` 非空则覆盖。没点名不要把库里仅有的那个值自动填上。
- 提示词不假定汽车领域。分类 / 改写 / 标签同样借 `chat_text`。标签候选函数写在 `classify.py`，逻辑上属意图层。

### 2.2 召回与过滤

先按 `filters.source` 强制筛选（空则该维不限）。`filters.tags` **不做强制筛选**：命中 N 条的块加 `min(N × TAG_BOOST_PER_HIT, TAG_BOOST_MAX)` 分；同一对只算一次。按字段强制筛选会把同字段多值切成 0 命中。

- 向量路：LanceDB 余弦近邻。来源可走 `where`。
- BM25 路：倒排；来源在 `source_matches` 筛。
- 两路 RRF 融合，截到候选窗口（`max(TOP_K × CANDIDATE_MULTIPLIER, CANDIDATE_FLOOR)`，默认 20）。不在这一层截 `TOP_K`。
- 词面兜底：embedding 调不通（`EMBED_UNAVAILABLE`）时不把异常抛给上层，退化为纯 BM25，命中标 `lexical_fallback=True`。向量服务挂掉不许把整个问答口打死（闲聊仍 200）。

调试题材由 `retrieve_hits(write_debug=)` 控制，默认取 `DEBUG_WRITE_QA`（默认关）。打开时写 `backend/data/result/qa/`，文件名带 session（只留 `[A-Za-z0-9_-]`、截到 `DEBUG_TAG_MAX_CHARS`）；评测与测试传 `False`。

### 2.3 重排

`rerank_hits` 把候选截成 `TOP_K`，保持 RRF 顺序只截断，不换序。截完后 `expand_hits_by_catalogue`：按目录标题把同节邻块并进命中 `text` / `pages`，有 token 上限；空标题不并；同标题去重。不要从截断前的候选里新增命中 id。文件仍叫 `overlap.py`。

### 2.4 生成与拒答

空命中拒答。`REFUSE_MIN_SCORE` 大于 0 时，最好的 `vector_score`（余弦）低于该值也拒。不看 RRF 融合分。词面兜底时另走一套：没有 `vector_score` 不等于有证据，`REFUSE_ON_LEXICAL_FALLBACK` 开着就要求命中里出现查询词。拒答文案 `REFUSE_TEXT`，`citations` 仍是列表（可空）。

提示词组装：检索命中拼进提示，另带文档身份 `tags`（与芯片同源）用来回答来源 / 品牌 / 车型。操作和规格只根据原文片段。程序组装 `citations[{id, source, pages, snippet}]`，模型只写 `answer`。检索门面把引用放侧栏，由答案旁按钮选定。

最小评测：正样本看 Hit@1 / Hit@`TOP_K`（期望来源，可再对页）；负样本看 `should_refuse`。`ROUTE_OVERRIDE="chitchat"` 注入后正样本必须全部未命中。

## 3. 字段约定

插件拆装只保证这些名字。其余进可选 metadata，不进约定。新入库写出这些字段；tags 可空，空则不加权。

**可分块原文** 必有：`source`，`kind`，以及 `pages[{page, text}]` 或无页时的 `text`。过滤标签：`tags[{field, value}]`，可 `[]`。不要把 `source` 写入 tags。

**知识块** 必有：`id`，`text`，`metadata.source`，`metadata.pages`（可 `[]`；块级页码）。过滤标签：`metadata.tags[{field, value}]`，可 `[]`，继承该来源可分块原文的 tags。可选不进约定：`catalogue_title`，`part`，`knowledge_type`。向量存在后端记录里，不进知识块约定。不要用 `metadata.brand` / `metadata.car` 当约定字段。

**KV 条目** 必有：`id`，`key`，`value`，`metadata.source`。可选：`metadata.tags`。

**查询** 必有：`text`。可选：`session`；`filters.source`（空串 = 不限来源，强制筛选）、`filters.tags`（空列表 = 不加权）仅调试覆盖意图。

**检索命中** 必有：`id`，`kind`（`chunk` 或 `kv`），`text`，`score`，`source`，`pages`。向量路过的命中另带 `vector_score`（余弦），只给拒答与评测，不进生成约定。

**生成结果** 必有：`answer`，`citations[{id, source, pages, snippet}]`，`tags[{field, value}]`（问句未点名则为 `[]`）。
