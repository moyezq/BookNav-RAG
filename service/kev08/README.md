# kev08 —— Kev-0.8B 的最小验证服务

权重：ModelScope `jaredpalmer/kev-0.8b` + 底座 `Qwen/Qwen3.5-0.8B-Base`｜上游思路：[jaredpalmer/kev](https://github.com/jaredpalmer/kev)
**本目录自给自足**：环境、权重、推理、HTTP 服务、探针都在这一层；不 import 上游 `kev` 包，不读外面任何解释器。

## 四步（都在 `service/kev08` 里执行）

```powershell
cd service/kev08
uv sync                                          # 建本目录 .venv（3.13），装 torch/transformers/peft/fastapi/uvicorn
uv run python download_kev08.py                  # 下权重：models/kev-0.8b + models/base
uv run python serve_kev08.py                     # 起服务 → http://127.0.0.1:8034（接口文档 /docs）
uv run python probe_kev08.py                     # 另开终端：输入 → 模型视图 → 原始响应 → 解析算式
```

没有 `.venv`、缺包、缺权重，一律直接报错终止；脚本不往别处找解释器，也没有 fallback。

## 谁的代码在跑

| 文件 | 职责 |
|---|---|
| `model_kev08.py` | 推理本体：编码、前向、指针头打分、温度校准、组响应 |
| `serve_kev08.py` | 只回 `POST /v1/systemone` 与 `GET /v1/models` 的 HTTP 服务（FastAPI + uvicorn） |
| `download_kev08.py` | ModelScope 下权重，并把底座摆成 kev 认的 HF 缓存布局（硬链接） |
| `probe_kev08.py` | 打印请求原文 / 模型看到的文本 / 响应原文 / 解析算式 |

推理链路（`model_kev08.py`，与上游 `kev/model.py` 同一套规则复刻）：

1. **编码**：state 以 `<|fim_prefix|>` 开头；每个问题一行 `<|fim_middle|>` 指令，然后每选项 `<|box_start|> 选项文本 <|box_end|>`，末尾 `<|fim_suffix|>`
2. **前向**：Qwen3.5 是 hybrid（Gated DeltaNet），不能吃块因果掩码，所以每个问题一行、行首带整段 state（row 形式）
3. **打分**：指针头把「各选项 `</opt>` 的隐状态」对「行末 `<|fim_suffix|>` 的隐状态」点积
4. **校准**：点积除以 `head.pt` 里的温度（0.8B = 2.35），softmax 得概率；`noul` 只回 `p(yes)`，`choice` 回 argmax + 每选项概率 + confidence

与上游 `kev.serve` 的差别，只有两处：这里**每个问题重算一遍 state**（上游用前缀缓存，两者按 fp32 舍入相等），并且**没实现 `score` 类型**。算法本身同源。

## 要 CUDA（有 3060 就别用 CPU 版 torch）

`uv sync` 默认从 PyPI 拿，Windows 上是 CPU 版。在 `pyproject.toml` 末尾加

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

再 `uv sync`。`serve_kev08.py` 的 `DEVICE = "auto"` 会自动选 cuda。`fla` / `triton` / `causal_conv1d` 在 Windows 没有轮子，不装也能跑（纯 PyTorch 参考实现，慢）。

## 要知道的

- 端口 **8034**；`probe_kev08.py` 默认打 8034，`KEV_BASE_URL` 可覆盖。
- 服务两条起法都行：`uv run python serve_kev08.py`（读文件里的 `HOST` / `PORT`），或 `uv run uvicorn serve_kev08:app --port 8034`。`/docs` 里能直接填一段 body 试请求。
- 权重在 `lifespan` 里加载，所以 `import serve_kev08` 本身不吃模型；不带权重的接口形状可以 `python -c "import serve_kev08; print(serve_kev08.app.openapi())"` 直接看。
- 前向是同步函数 + 一把锁：并发请求在 Starlette 线程池里排队，不会同时抢显存。
- 出错响应体是 FastAPI 的 `{"detail": ...}`（手写版曾是 `{"error": {"message": ...}}`），TypeSafe 的 SDK 认这个形状。
- 精度固定 **fp32**：这是上游"报告数字"的那条路径；GPU 上想省显存再改 `resolve_device` 里的 dtype。
- 权重不进仓：`models/`、`.venv/`、`uv.lock` 都在 `.gitignore` 里。
- 0.8B 的边界：模型卡语言是 English，验证上下文 8k（服务上限 65,536），工具路由 When2Call 0.133 低于瞎猜。
