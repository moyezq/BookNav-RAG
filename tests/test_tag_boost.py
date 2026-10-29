# -*- coding: utf-8 -*-
"""
过滤标签加权（ADR-0035）：命中标签的块上浮，没命中的块不许被丢掉。

职责：锁定三件事——
      1. `_rrf_fuse` 的加权：命中 N 条标签加 N × TAG_BOOST_PER_HIT，封顶 TAG_BOOST_MAX，0 条不加；
      2. `KnowledgeBase.search`：标签不再强制筛选（未命中标签的块照样留在候选里），
         命中标签的块能靠加权排到前面；
      3. 单文档库上的恒等性：所有块标签相同时加权对所有候选一视同仁 → 排序不变
         （所以本项在只有一份手册时不会有指标变化）。
      为什么要有这条：强制筛选时同字段两个值必然 0 命中 → 空命中 → 直接输出拒答文案，
      用户看不出是筛选把候选筛空的；改判据前这里先把行为写死。

不调 8030、不读真库：假向量库 + 空 query_text（只走向量路，把加权效果隔离出来）。

应用场景：改召回加权或过滤逻辑之后跑 python -m pytest tests/test_tag_boost.py。
"""

from __future__ import annotations

import pytest

from backend.conf import TAG_BOOST_MAX
from backend.offline.vector.store import Bm25Index
from backend.online.filter.tags import empty_filters
from backend.online.recall.retrieve import KnowledgeBase, _rrf_fuse

QUERY_VECTOR = [1.0, 0.0]


def _record(record_id: str, source: str, text: str, tags: list) -> dict:
    """一条够 search / _hit_from_record 用的库记录。"""
    return {
        "id": record_id,
        "text": text,
        "metadata": {"source": source, "pages": [1], "tags": tags, "catalogue_title": ""},
        "source": source,
        "pages": [1],
    }


def _hit(record: dict, score: float) -> dict:
    """假向量路返回的命中（形状与 _hit_from_record 一致）。"""
    return {
        "id": record["id"],
        "kind": "chunk",
        "text": record["text"],
        "score": score,
        "source": record["source"],
        "pages": record["pages"],
    }


class _FakeVectorStore:
    """按给定顺序回放命中；load_payload 用来把知识库装起来。"""

    def __init__(self, records: list, order: list):
        self._records = records
        self._order = order

    def load_payload(self):
        return self._records, "bge-m3", 2

    def search_vectors(self, query_vector, filters, limit):
        by_id = {record["id"]: record for record in self._records}
        return [
            _hit(by_id[record_id], 1.0 - index * 0.01)
            for index, record_id in enumerate(self._order[:limit])
        ]


def _build_knowledge_base(records: list, order: list) -> KnowledgeBase:
    """两个块、两个来源的知识库；向量路按 order 回放（order 第一条是头名）。"""
    knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
    knowledge_base.lancedb_dir = None
    knowledge_base.vector_store = _FakeVectorStore(records, order)
    knowledge_base.embedder_name = "bge-m3"
    knowledge_base.dim = 2
    knowledge_base.records = records
    knowledge_base.bm25 = Bm25Index()
    knowledge_base.bm25.build([(record["id"], record["text"]) for record in records])
    return knowledge_base


def _tags(*pairs) -> list:
    """按 field=value 造标签列表。"""
    return [{"field": field, "value": value} for field, value in pairs]


def test_加权让命中标签的块上浮():
    """B 本是向量路头名；A 命中一条标签 → A 靠加权反超（0.005 远大于 RRF 相邻名次差）。"""
    hits = _rrf_fuse(
        [
            {"id": "B", "kind": "chunk", "text": "b", "score": 0.9, "source": "s", "pages": []},
            {"id": "A", "kind": "chunk", "text": "a", "score": 0.8, "source": "s", "pages": []},
        ],
        [],
        5,
        {"A": 1, "B": 0},
    )

    assert [hit["id"] for hit in hits] == ["A", "B"]


def test_加权封顶():
    """命中 10 条也只加 TAG_BOOST_MAX。"""
    hits = _rrf_fuse(
        [{"id": "A", "kind": "chunk", "text": "a", "score": 0.9, "source": "s", "pages": []}],
        [],
        5,
        {"A": 10},
    )

    assert hits[0]["score"] == pytest.approx(1.0 / 61 + TAG_BOOST_MAX)


def test_零命中与库外id都不影响分数():
    """matched=0 或不在这批候选里 → 不加分，也不许报错。"""
    hits = _rrf_fuse(
        [{"id": "A", "kind": "chunk", "text": "a", "score": 0.9, "source": "s", "pages": []}],
        [],
        5,
        {"A": 0, "库外": 3},
    )

    assert hits[0]["score"] == pytest.approx(1.0 / 61)


def test_标签不再把块筛掉且命中标签的排前():
    """旧行为：B 不含「主题=充电」会被整条丢掉；现在两条都在，A 上浮到第一。"""
    a = _record("A", "手册A", "充电说明：慢充与快充。", _tags(("主题", "充电")))
    b = _record("B", "手册B", "保养周期：每 10000 公里。", _tags(("主题", "保养")))
    knowledge_base = _build_knowledge_base([a, b], ["B", "A"])
    filters = {"source": "", "tags": _tags(("主题", "充电"))}

    hits = knowledge_base.search(QUERY_VECTOR, filters, 5, query_text="")

    assert [hit["id"] for hit in hits] == ["A", "B"]


def test_同字段两个值不再筛成零命中():
    """同字段两值过去必然 0 命中 → 拒答；现在两条各加一分、都留着。"""
    a = _record("A", "手册A", "充电说明。", _tags(("主题", "充电")))
    b = _record("B", "手册B", "保养周期。", _tags(("主题", "保养")))
    knowledge_base = _build_knowledge_base([a, b], ["B", "A"])
    filters = {"source": "", "tags": _tags(("主题", "充电"), ("主题", "保养"))}

    hits = knowledge_base.search(QUERY_VECTOR, filters, 5, query_text="")

    assert len(hits) == 2
    assert [hit["id"] for hit in hits] == ["B", "A"]


def test_所有块的标签一样时加权不改变顺序():
    """单文档库的恒等性：每块都带同一份标签 → 加分相同 → 顺序就是召回顺序（本轮无指标变化）。"""
    a = _record("A", "手册", "充电说明。", _tags(("主题", "充电"), ("品牌", "比亚迪")))
    b = _record("B", "手册", "保养周期。", _tags(("主题", "充电"), ("品牌", "比亚迪")))
    knowledge_base = _build_knowledge_base([a, b], ["B", "A"])

    plain = knowledge_base.search(QUERY_VECTOR, empty_filters(), 5, query_text="")
    boosted = knowledge_base.search(
        QUERY_VECTOR, {"source": "", "tags": _tags(("主题", "充电"))}, 5, query_text=""
    )

    assert [hit["id"] for hit in plain] == ["B", "A"]
    assert [hit["id"] for hit in boosted] == ["B", "A"]
    assert boosted[0]["score"] > plain[0]["score"]


def test_来源仍然强制筛选():
    """来源是文档身份：点名手册A 时手册B 的块不许进候选。"""
    a = _record("A", "手册A", "充电说明。", _tags(("主题", "充电")))
    b = _record("B", "手册B", "保养周期。", _tags(("主题", "保养")))
    knowledge_base = _build_knowledge_base([a, b], ["B", "A"])

    hits = knowledge_base.search(QUERY_VECTOR, {"source": "手册A", "tags": []}, 5, query_text="")

    assert [hit["id"] for hit in hits] == ["A"]
