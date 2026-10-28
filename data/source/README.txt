本目录放待入库的 PDF。.gitignore 默认忽略这里的文件，只放行本说明和仓库自带的演示手册。

放什么：
- 仓库自带演示：`demo_handbook.pdf`（自造 6 页短手册，可再分发；若缺失、过大或要改文案，在仓库根执行 python scripts/ingest/make_demo_pdf.py）。
- 其它自造或可再分发文档（公有领域、CC 授权、你自己写的）。
- 第三方版权文档（厂商手册、付费标准、内部资料）只在本机跑，不要提交、也不要压进发布包。

入库：文件放这里后，先起 `python -m embed.server`，再 `python scripts/ingest/ingest.py`。
本阶段只支持 PDF。
