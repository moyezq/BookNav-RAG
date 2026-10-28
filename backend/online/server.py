# -*- coding: utf-8 -*-
"""
问答 HTTP 进程入口。

职责：绑定本文件 HOST/PORT，挂 backend.online.router，注册 lifespan。不是管理台。
前置：先起 `python -m service.embed.server`；知识库 JSON 已入库。
对外：POST /api/ask 在 backend/online/router.py。GET / 探活。端口与 frontend/vite.config.ts proxy 耦合。
怎么跑：在仓库根 `python -m backend.online.server`。不要 `python backend/online/server.py`。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import backend.conf as conf  # ROOT 进 path；读 yaml 与 .env
import backend.online.router as ask_http
from backend.online.recall.retrieve import open_knowledge_base

# ========== 按需修改 ==========
HOST = "127.0.0.1"
PORT = 8031
# ==============================


class _SkipPortProbeAccessLog(logging.Filter):
    """编辑器在端口起来后会探 GET / 与 GET /json/version，不是问答请求。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """这两条探测不进 uvicorn access 日志。"""
        message = record.getMessage()
        if "/json/version" in message:
            return False
        if '"GET / HTTP' in message:
            return False
        return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时读入向量库 JSON，进程退出时不额外清理。"""
    logging.getLogger("uvicorn.access").addFilter(_SkipPortProbeAccessLog())
    ask_http.knowledge_base = open_knowledge_base()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(ask_http.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
