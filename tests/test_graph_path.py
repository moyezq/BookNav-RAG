# -*- coding: utf-8 -*-
"""
在线检索核心的单元测试（不调 LLM、不调向量服务）。

职责：锁定 backend/online/graph.py 里两件与「评测到底测了什么」直接相关的事——
      1. 过滤对象归一化：dict 与 HTTP 层的 Pydantic 对象都要认（这是 O5 那类静默出错的来源）；
      2. ask_once_core 的闲聊分支必须在召回之前返回：注入 chitchat 时不许碰知识库。
         这一条就是「注入 chitchat 后评测必须失败」的机内版本。

应用场景：改 graph.py 之后跑 python -m pytest tests/test_graph_path.py。
"""

from __future__ import annotations

from backend.online.graph import _normalize_filters, ask_once_core
from backend.online.schema import AskFilters, FilterTag


class _ExplodingKnowledgeBase:
    """任何被调用都失败的替身：用来证明闲聊分支真的没碰检索。"""

    def tag_catalog(self):
        raise AssertionError("闲聊不该读标签目录")

    def search(self, *args, **kwargs):
        raise AssertionError("闲聊不该召回到向量库")

    def expand_hits_by_catalogue(self, hits):
        raise AssertionError("闲聊不该并同节邻块")


# ========== 过滤对象归一化 ==========


def test_归一化认dict():
    normalized = _normalize_filters({"source": "手册A", "tags": [{"field": "品牌", "value": "比亚迪"}]})

    assert normalized == {"source": "手册A", "tags": [{"field": "品牌", "value": "比亚迪"}]}


def test_归一化认HTTP层的模型对象():
    """router 传下来的是 Pydantic 对象，不是 dict。"""
    body_filters = AskFilters(source="手册A", tags=[FilterTag(field="品牌", value="比亚迪")])
    normalized = _normalize_filters(body_filters)

    assert normalized == {"source": "手册A", "tags": [{"field": "品牌", "value": "比亚迪"}]}


def test_归一化丢掉空项与半条标签():
    normalized = _normalize_filters(
        {"source": "  ", "tags": [{"field": "", "value": "x"}, {"field": "品牌", "value": ""}]}
    )

    assert normalized == {"source": "", "tags": []}


def test_归一化对None给空过滤():
    assert _normalize_filters(None) == {"source": "", "tags": []}


# ========== 闲聊分支不碰检索 ==========


def test_注入闲聊时在召回之前返回():
    """route_override=chitchat 必须直接返回空命中，一次都不许碰知识库。"""
    result = ask_once_core(
        _ExplodingKnowledgeBase(),
        "后视镜怎么折叠",
        route_override="chitchat",
        write_debug=False,
    )

    assert result["route"] == "chitchat"
    assert result["hits"] == []
    assert result["filters"] == {"source": "", "tags": []}
    assert result["rewrite_failed"] is False


def test_注入闲聊时不改写不抽标签():
    """闲聊分支的 text 就是用户原文。"""
    result = ask_once_core(
        _ExplodingKnowledgeBase(),
        "你好",
        route_override="chitchat",
        write_debug=False,
    )

    assert result["text"] == "你好"
