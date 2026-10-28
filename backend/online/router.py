# -*- coding: utf-8 -*-
"""
问答 HTTP 路由。

职责：POST /api/ask。不是进程入口，没有 __main__。
前置：backend/online/server.py 在 lifespan 里 open_knowledge_base。
对外：router；AskBody / Generation 字段见 schema.py。GET / 探活。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.online.graph import _normalize_filters, run_ask_graph
from backend.online.schema import AskBody, GenerationResult

router = APIRouter()

# backend.online.server lifespan 写入；提问前必须已 load
knowledge_base = None


@router.get("/")
def root() -> dict:
    """
    探活。

    功能：告诉调用方服务已起；问答仍是 POST /api/ask。
    应用场景：打开 8031 根路径；编辑器端口探测。
    输入：无。
    输出：{ok, ask, docs}。
    """
    return {"ok": True, "ask": "/api/ask", "docs": "/docs"}


def _request_filters(body: AskBody) -> dict | None:
    """有任一过滤字段才覆盖意图标签；全空则不覆盖。与图入口共用同一套归一化。"""
    normalized = _normalize_filters(body.filters)
    if normalized["source"] or normalized["tags"]:
        return normalized
    return None


@router.post("/api/ask", response_model=GenerationResult)
def ask(body: AskBody) -> dict:
    """
    检索并生成带引用的答案。

    功能：调用在线图 intent →（闲聊则跳过召回）recall → rerank → generate。
    应用场景：问答页提问。
    输入：body.text — 查询；session 可空；filters 非空则覆盖意图标签。
    输出：{answer, citations, tags}。拒答开且证据不足时仍 200，citations 可空。
    """
    question = body.text.strip()
    if not question:
        raise HTTPException(status_code=400, detail="text 不能为空")
    if knowledge_base is None:
        raise HTTPException(status_code=503, detail="知识库未加载")
    try:
        return run_ask_graph(
            knowledge_base,
            question,
            _request_filters(body),
            body.session.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
