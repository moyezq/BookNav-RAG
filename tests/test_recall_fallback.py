# -*- coding: utf-8 -*-
"""
词面兜底的单元测试：向量服务不可用时不许把整次提问打死。

职责：锁定三件事——
      1. embedding 调用失败时，召回退化为纯 BM25，并且命中带 lexical_fallback=True；
      2. 兜底命中的拒答不走「没有 vector_score 就放行」，改看词面是否有重合；
      3. 库内记录与 BM25 倒排不因为降级而受损。

不依赖真实向量服务：用一个指向必失败地址（端口 1）的 HttpEmbedder，等价于 8030 挂掉。

应用场景：改召回降级或拒答逻辑之后跑 python -m pytest tests/test_recall_fallback.py。
"""

from __future__ import annotations

import pytest

from backend.online.generate.answer import hits_contain_query_term, should_refuse
from backend.online.recall.retrieve import EMBED_UNAVAILABLE, KnowledgeBase, retrieve_hits
from backend.online.recall.retrieve import _build_embedder  # noqa: F401  — 说明降级用的是同一个构造器

# 指向必然连不上的地址：本机没有服务监听 1 端口
DEAD_EMBED_URL = "http://127.0.0.1:1/v1/embeddings"


class _FakeVectorStore:
    """只提供 load_payload 的最小替身，用来把知识库装起来（不读向量列）。"""

    def __init__(self, records):
        self._records = records

    def load_payload(self):
        return self._records, "bge-m3", 1024


def _build_knowledge_base() -> KnowledgeBase:
    """造一份两个块的知识库：一个讲后视镜，一个讲洗涤液。"""
    records = [
        {
            "id": "手册-0001",
            "text": "外后视镜可以电动折叠，锁车时自动收起。",
            "metadata": {"source": "手册", "pages": [93], "tags": [], "catalogue_title": "外后视镜"},
            "source": "手册",
            "pages": [93],
        },
        {
            "id": "手册-0002",
            "text": "洗涤液不足时，仪表会提示加注玻璃水。",
            "metadata": {"source": "手册", "pages": [360], "tags": [], "catalogue_title": "洗涤液"},
            "source": "手册",
            "pages": [360],
        },
    ]
    knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
    knowledge_base.lancedb_dir = None
    knowledge_base.vector_store = _FakeVectorStore(records)
    knowledge_base.embedder_name = "bge-m3"
    knowledge_base.dim = 1024
    knowledge_base.records = records
    from backend.offline.vector.store import Bm25Index

    knowledge_base.bm25 = Bm25Index()
    knowledge_base.bm25.build([(record["id"], record["text"]) for record in records])
    return knowledge_base


@pytest.fixture()
def dead_embed_url(monkeypatch):
    """把 embedding 地址换成必失败地址，等价于 8030 挂掉。"""
    monkeypatch.setattr("backend.online.recall.retrieve.EMBED_API_URL", DEAD_EMBED_URL)


def test_向量服务不可用时退化为纯BM25(dead_embed_url):
    """embedding 连不上时仍拿得到命中，且命中被标为词面兜底。"""
    knowledge_base = _build_knowledge_base()
    hits = retrieve_hits(knowledge_base, "后视镜怎么折叠", {"source": "", "tags": []}, write_debug=False)

    assert hits, "兜底必须还能召回，不能直接空手"
    assert all(hit.get("lexical_fallback") for hit in hits)
    assert hits[0]["source"] == "手册"


def test_异常类型被降级捕获(dead_embed_url):
    """连不上的异常必须落进 EMBED_UNAVAILABLE，否则会穿透成 500。"""
    import requests

    with pytest.raises(EMBED_UNAVAILABLE):
        requests.post(DEAD_EMBED_URL, json={"model": "x", "input": ["a"]}, timeout=2)


def test_兜底命中不破坏知识库(dead_embed_url):
    """降级只改查询方式，不改库内记录与倒排。"""
    knowledge_base = _build_knowledge_base()
    before = len(knowledge_base.records)

    retrieve_hits(knowledge_base, "后视镜怎么折叠", {"source": "", "tags": []}, write_debug=False)

    assert len(knowledge_base.records) == before
    assert knowledge_base.bm25.n_docs == before


def test_兜底命中按词面拒答():
    """没有词面重合就拒答；有重合才放行。"""
    unrelated = [{"id": "a", "text": "红烧肉要先焯水。", "lexical_fallback": True}]
    related = [{"id": "b", "text": "后视镜可以电动折叠。", "lexical_fallback": True}]

    assert hits_contain_query_term(unrelated, "后视镜怎么折叠") is False
    assert should_refuse(unrelated, "后视镜怎么折叠") is True
    assert should_refuse(related, "后视镜怎么折叠") is False


def test_普通向量命中不受词面兜底开关影响():
    """有向量分的命中仍走余弦阈值。"""
    cosine_high = [{"id": "a", "text": "无关内容", "vector_score": 0.9}]
    cosine_low = [{"id": "b", "text": "无关内容", "vector_score": 0.1}]

    assert should_refuse(cosine_high, "任何问句") is False
    assert should_refuse(cosine_low, "任何问句") is True
