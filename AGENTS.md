# 给 AI 的本仓约定

全局 skill 已规定的写法、怎么跑、todo、测试夹具、文档地图，不在这里复述。只放本仓例外，以及反复要用的检查落在仓内哪一处。

本仓多出来的文档：[docs/compare.md](docs/compare.md)（产品仓对照）、[docs/modelselection.md](docs/modelselection.md)（模型结论）。问答页视觉规范在 `frontend/DESIGN.md`，不在仓库根。

## 禁令

- 端口从 8030 连号：向量 8030、问答 8031、Vite 8032、预备 kev08 8034。不要写回 8081 / 8090 / 5173 / 8080。
- `service/kev08/` 用它自己的 `pyproject.toml` 与 `.venv`。不要把 kev 的 torch 装进本仓环境，不要 `import` 进提问口，不要塞进 `start.py`。
- 只读本仓 `.env`：`backend.conf.apply_env()` 是唯一读取点。不要读别的仓的 `.env`。
- 产品服务从仓库根 `uv run python start.py`，或 `uv run python -m service.embed.server` / `uv run python -m backend.online.server`。不要 `python service/embed/server.py` 或 `python backend/online/server.py`（`sys.path` 会指到子目录）。
- 本机验证外部模型：先给方案（官方文档、权重仓库与文件、下载命令、最小验证请求），等人说可以再下。权重不自己下。torch 这类 GB 级依赖装之前必须问；已有 GPU 环境先确认能否复用，不要默认装 CPU 版。GitHub 直连超时或 5 分钟没有字节落地就停，改 tarball / 镜像或请用户给。一步卡住超过 10 分钟停手汇报。
- 不要重建 `query/`。离线不要再拆独立的分块层 / 向量化层 / 向量存储层。向量模型服务进 `service/embed/`，不要摊在 `service/` 根上，也不要放进 `backend/offline/` 或 `backend/online/`。向量库不要写回每来源一份 `step5` JSON。
- 层与模块开关只改 `backend/conf/base.py`。不把 ROOT、host、port、第三方地址、密钥写进 `backend/conf/config.yaml`。向量权重目录是 `service/embed/models/`，不写进 yaml。
- 不要重建 `extract/methods/`。向量层不要 import 抽取层。不要拆共享 Chat 客户端。
- 不把产品仓的模块名、类名、`parser_id` 写进本仓代码或术语表。不 fork 产品仓。不把 anyurl2text、MinerU、anydoc 源码引进本仓。不要再接外仓 `/rag/extract/file`。
- 提问口不接 `create_agent` / `create_deep_agent`、词重叠重排。不要做 KV 存储或 KV 召回、图片 OCR、ask 热加载。

## 检查落点

- 入库、评测、全量 pytest 后台起；中止后核对有没有留下 python 残留进程。
- pytest 到点自退：`pyproject.toml` 的 `[tool.pytest.ini_options]`（`faulthandler_timeout`）。写盘用例：`tests/conftest.py` 的 `work_dir`。
