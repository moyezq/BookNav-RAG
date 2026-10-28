"""
本目录自己的 Kev-0.8B 服务：FastAPI 只回 POST /v1/systemone 与 GET /v1/models。

职责：加载 models/ 下的权重，用 model_kev08.py 的推理实现回答请求；不 import kev、不调用它的 serve。
前置：在本目录执行过 uv sync（.venv 里要有 torch / transformers / peft / fastapi / uvicorn）与 download_kev08.py（models/ 下有权重）。
      缺解释器、缺包、缺权重都直接让它报错，不往别处找退路。
对外：POST /v1/systemone（TypeSafe 形状，支持 noul / choice）、GET /v1/models（本进程的运行参数）、
      GET /docs（自动生成的接口文档，可直接点「Try it out」发一次请求）。
怎么跑：uv run python serve_kev08.py
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, model_validator

from model_kev08 import MAX_STATE, Model, answer, load_model

# ========== 按需修改 ==========
HOST = "127.0.0.1"
PORT = 8034  # 本仓决策服务端口位，接 8030～8033 连号
DEVICE = "auto"  # auto | cpu | cuda
# ==============================

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "models" / "kev-0.8b"
BASE_DIR = HERE / "models" / "base"

_LOCK = threading.Lock()  # 一次只跑一个前向，避免并发抢显存/内存
logger = logging.getLogger("uvicorn.error")  # 借用 uvicorn 已配好的日志器，不另建 logging 配置


class Question(BaseModel):
    """TypeSafe 的一个问题：noul 回 p(true)，choice 回各选项概率。"""

    type: Literal["noul", "choice"]  # 上游还有 score，本实现不处理，这里直接拒成 422
    instructions: str | None = None
    criteria: dict[str, Any] | None = None  # choice 必填；noul 用 true / false 两个键当选项文案

    @model_validator(mode="after")
    def require_criteria(self) -> Question:
        """choice 没有 criteria 就没有选项，放到下游会变成 500，这里先拦成 422。"""
        if self.type == "choice" and not self.criteria:
            raise ValueError("choice 类型必须带 criteria（选项名 → 选项描述）")
        return self


class SystemOneRequest(BaseModel):
    """POST /v1/systemone 的请求体，字段名沿用 TypeSafe。"""

    state: str
    questions: dict[str, Question]
    model: str = "kev-latest"


def resolve_device(requested: str) -> tuple[str, torch.dtype]:
    """把 DEVICE 常量翻译成 (设备, 精度)：没卡就用 CPU fp32。"""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda 但没有可用 GPU；改成 auto 或 cpu")
    return requested, torch.float32


def get_model(request: Request) -> Model:
    """取 lifespan 里加载好的模型；没加载就抛错（不静默返回 None）。"""
    model: Model | None = getattr(request.app.state, "model", None)
    if model is None:
        raise RuntimeError("模型尚未加载")
    return model


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """启动时加载一次权重，退出时丢掉引用；load 放在这里，import 本模块就不会吃掉整个模型。"""
    device, dtype = resolve_device(DEVICE)
    on_cuda = device == "cuda"
    logger.info("载入 %s  +  底座 %s  device=%s dtype=%s", RUN_DIR, BASE_DIR, device, str(dtype).replace("torch.", ""))
    application.state.model = load_model(RUN_DIR, BASE_DIR, device=device, dtype=dtype)
    logger.info(
        "温度 %s  head_dim %s  cuda=%s  state 上限 %s tokens",
        application.state.model.head.temperature,
        application.state.model.head.q.out_features,
        on_cuda,
        f"{MAX_STATE:,}",
    )
    try:
        yield
    finally:
        application.state.model = None


app = FastAPI(title="kev08", lifespan=lifespan)


@app.get("/v1/models")
def list_models(request: Request) -> dict[str, Any]:
    """GET /v1/models：把本进程实际用的权重、设备、精度、温度报出来。"""
    model = get_model(request)
    return {
        "models": [
            {
                "name": "kev-latest",
                "run": str(model.run_dir),
                "base": str(BASE_DIR),
                "device": model.device,
                "dtype": str(model.dtype).replace("torch.", ""),
                "backend": "torch",
                "temperature": model.head.temperature,
                "head_dim": model.head.q.out_features,
                "max_state_tokens": MAX_STATE,
                "note": "本目录自实现的最小服务，未实现 score 类型与 state 前缀缓存",
            }
        ]
    }


@app.post("/v1/systemone")
def systemone(body: SystemOneRequest, request: Request) -> dict[str, Any]:
    """
    POST /v1/systemone：读 state 与 questions，回 TypeSafe 形状的 answers。

    五步：取模型 → 摊平问题 → 前向打分 → 数输出 token → 组响应，每步前面有一行带步号的注释。
    写成同步函数是有意的：一次前向要几秒，async 会把事件循环整个卡住；Starlette 会把同步路由
    丢进线程池，再由 _LOCK 串起来，并发请求排队而不是抢显存。
    """
    # 1. 取模型：lifespan 里加载好的那个，没加载会抛 RuntimeError
    model = get_model(request)

    # 2. 摊平问题：Pydantic 模型 → 推理层吃的普通 dict，没填的字段不往下传
    questions = {}
    for name, question in body.questions.items():
        questions[name] = question.model_dump(exclude_none=True)

    # 3. 前向打分：加锁串行；state 超长这类 ValueError 统一转 422
    try:
        with _LOCK:
            answers, input_tokens, elapsed = answer(model, body.state, questions)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # 4. 数输出 token：模型只回概率不吐文本，只能把响应体再分词一遍
    response_text = json.dumps(answers)
    output_tokens = len(model.tok(response_text, add_special_tokens=False).input_ids)

    # 5. 组响应：TypeSafe 形状，字段名与上游 kev.serve 一致
    return {
        "model": body.model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "latency_ms": round(elapsed * 1000, 1),
    }    


if __name__ == "__main__":
    """起 uvicorn 并阻塞到 Ctrl-C；用 app 对象而不是 import 字符串，不启用 reload。"""
    uvicorn.run(app, host=HOST, port=PORT)
