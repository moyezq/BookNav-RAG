# -*- coding: utf-8 -*-
"""
按来源替换写入不得「先删光再失败」。

职责：锁定 LanceVectorStore.replace_source —— 暂存 add 失败时旧正式来源行数不变。
"""

from __future__ import annotations

import pytest

from backend.offline.vector.store import WRITING_MARKER, LanceVectorStore


def _chunk(chunk_id: str, text: str = "旧正文。") -> dict:
    """一个够 _rows_from_chunks 用的知识块。"""
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source": "手册",
            "pages": [1],
            "tags": [],
            "catalogue_title": "第一节",
        },
    }


def test_暂存add失败时旧来源仍在(work_dir, monkeypatch):
    """模拟写入暂存行失败：正式来源一行都不能少。"""
    store = LanceVectorStore(work_dir / "db")
    store.replace_source("手册", [_chunk("手册-0001")], [[1.0, 0.0]], "stub", 2)
    assert store.count_source_rows("手册") == 1

    def boom(self, table, rows):
        raise RuntimeError("simulated add failure")

    monkeypatch.setattr(LanceVectorStore, "_add_rows", boom)

    with pytest.raises(RuntimeError, match="simulated add failure"):
        store.replace_source(
            "手册",
            [_chunk("手册-0001", "新正文。")],
            [[0.0, 1.0]],
            "stub",
            2,
        )

    assert store.count_source_rows("手册") == 1
    assert store.count_source_rows(f"手册{WRITING_MARKER}") == 0


def test_正常替换后没有暂存行(work_dir):
    """走完全程后正式来源更新，暂存来源为 0。"""
    store = LanceVectorStore(work_dir / "db")
    store.replace_source("手册", [_chunk("手册-0001")], [[1.0, 0.0]], "stub", 2)
    store.replace_source(
        "手册",
        [_chunk("手册-0001", "新正文。"), _chunk("手册-0002", "第二块。")],
        [[0.0, 1.0], [1.0, 1.0]],
        "stub",
        2,
    )
    assert store.count_source_rows("手册") == 2
    assert store.list_writing_sources() == []
    records, _, _ = store.load_payload()
    assert len(records) == 2
    assert records[0]["text"] == "新正文。" or records[1]["text"] == "新正文。"
