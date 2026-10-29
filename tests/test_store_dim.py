# -*- coding: utf-8 -*-
"""
首次建表的维数校验：错维数抢跑一次，整库就锁死。

职责：锁定 backend/offline/vector/store.py::LanceVectorStore.replace_source 的两件事——
      1. 首次建表（表还不存在）时，声明维数与实际向量长度不符 → ValueError，且不留半截表；
      2. 已经建过 256 维的表之后再写 1024 维 → ValueError 同时报出两个维数（指向「删库重入库」）。
      为什么要有这条：create_table 分支以前完全不校验，错维数抢跑建表后，
      bge-m3 的 1024 维再也写不进去，只能删 backend/data/lancedb 全量重入。
      真库、真 LanceDB，需要写文件的目录用 work_dir（沙箱里 tmp_path 用不了）。

应用场景：改 replace_source 的建表分支之后跑本文件。
"""

from __future__ import annotations

import pytest

from backend.offline.vector.store import LanceVectorStore


def _chunk(chunk_id: str) -> dict:
    """一个够 _rows_from_chunks 用的知识块。"""
    return {
        "id": chunk_id,
        "text": "后视镜可以电动折叠。",
        "metadata": {
            "source": "手册",
            "pages": [93],
            "tags": [],
            "catalogue_title": "外后视镜",
        },
    }


def test_首次建表时声明维数与向量不符就报错(work_dir):
    """声明 4 维、向量 2 维：建表前就得拦，别把错维度的表建出来。"""
    store = LanceVectorStore(work_dir / "db")
    chunks = [_chunk("手册-0001")]

    with pytest.raises(ValueError) as excinfo:
        store.replace_source("手册", chunks, [[1.0, 0.0]], "stub", 4)

    assert "维数" in str(excinfo.value)
    assert store.db.table_names() == []


def test_向量长度不一致就报错(work_dir):
    """同一批里混进不同长度的向量：真因在调用方，不该等 LanceDB 抛难懂的错。"""
    store = LanceVectorStore(work_dir / "db")
    chunks = [_chunk("手册-0001"), _chunk("手册-0002")]

    with pytest.raises(ValueError) as excinfo:
        store.replace_source("手册", chunks, [[1.0, 0.0], [1.0, 0.0, 0.0]], "stub", 2)

    assert "长度不一致" in str(excinfo.value)


def test_先256维后1024维时报出两个维数(work_dir):
    """错维数抢跑的真实代价：第二次写入必须说清「表内 256、本次 1024、删库重入」。"""
    store = LanceVectorStore(work_dir / "db")
    chunks = [_chunk("手册-0001")]
    store.replace_source("手册", chunks, [[0.0] * 256], "stub", 256)

    with pytest.raises(ValueError) as excinfo:
        store.replace_source("手册", chunks, [[0.0] * 1024], "bge-m3", 1024)

    message = str(excinfo.value)
    assert "256" in message
    assert "1024" in message
