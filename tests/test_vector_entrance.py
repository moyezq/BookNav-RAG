# -*- coding: utf-8 -*-
"""
向量层入口的写进磁盘顺序：step4 是向量库的辅助证据，必须写库成功之后才写。

职责：锁定 backend/offline/vector/entrance.py::run_vector 的三件事——
      1. embed 失败 → 抛错，且 step4_chunks.json **不写进磁盘**（库里还是旧的，不许留半成品）；
      2. 写库成功时才写 step4，写进磁盘那一刻库里已经有这批块；
      3. 写进磁盘的 step4 与写进库的就是同一批块。
      为什么要有这条约定：2026-10-08 的入库在 embed 阶段 ReadTimeout，
      step4 已经是新的 138 块而 backend/data/lancedb 还是旧的 137 行、78 块页码不同；
      在线检索读的是库，于是「分块改好了」这句话在线上并不成立。
      不调模型、不写真库：替换 entrance.make_embedder 与 entrance.LanceVectorStore。

应用场景：改 run_vector 的写进磁盘顺序之后跑 python -m pytest tests/test_vector_entrance.py。
"""

from __future__ import annotations

import json

import pytest

from backend.offline.vector import entrance

SOURCE_OBJ = {
    "source": "样本",
    "kind": "pdf",
    "pages": [{"page": 1, "text": "后视镜可以电动折叠。"}, {"page": 2, "text": "洗涤液不足时仪表会提示。"}],
    "catalogue_ranges": [{"title": "某节", "start_page": 1, "end_page": 2}],
    "tags": [],
}


def _write_source(work_dir):
    """把可分块原文写到临时目录，返回文件路径（step4 会落在同一个目录）。"""
    path = work_dir / "chunkable_source.json"
    path.write_text(json.dumps(SOURCE_OBJ, ensure_ascii=False), encoding="utf-8")
    return path


class _FakeEmbedder:
    """固定维向量；name / dim 与 HttpEmbedder 同形，够 run_vector 用。"""

    name = "fake"
    dim = 2

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _ExplodingEmbedder:
    """embed 必然失败：等价于 8030 读超时/连不上。"""

    name = "fake"
    dim = 2

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed 超时")


@pytest.fixture()
def fake_store(monkeypatch) -> dict:
    """替身向量库：记录写库时的现场，包括「那一刻 step4 在不在」。"""
    recorded: dict = {}

    class FakeStore:
        def __init__(self, db_dir):
            self.db_dir = db_dir

        def replace_source(self, source, chunks, vectors, embedder_name, dim) -> None:
            recorded["source"] = source
            recorded["ids"] = [chunk["id"] for chunk in chunks]
            recorded["step4_exists_at_write"] = recorded["step4_path"].exists()

    monkeypatch.setattr(entrance, "LanceVectorStore", FakeStore)
    return recorded


def test_embed失败时不写step4(work_dir, monkeypatch, fake_store):
    """embed 挂了就整批作废：step4 不许变成新的，否则库旧、文件新，没人看得出来。"""
    fake_store["step4_path"] = work_dir / "step4_chunks.json"
    monkeypatch.setattr(entrance, "make_embedder", lambda: _ExplodingEmbedder())

    with pytest.raises(RuntimeError):
        entrance.run_vector(_write_source(work_dir))

    assert not (work_dir / "step4_chunks.json").exists()


def test_写库成功后才写step4(work_dir, monkeypatch, fake_store):
    """写进磁盘的顺序：写库在前、step4 在后。写库那一刻 step4 必须还不存在。"""
    step4_path = work_dir / "step4_chunks.json"
    fake_store["step4_path"] = step4_path
    monkeypatch.setattr(entrance, "make_embedder", lambda: _FakeEmbedder())

    entrance.run_vector(_write_source(work_dir))

    assert fake_store["step4_exists_at_write"] is False
    assert step4_path.is_file()


def test_写进磁盘的step4就是写进库的那批块(work_dir, monkeypatch, fake_store):
    """两个产出的块 id 必须一致：一份是检索读的，一份是调试看的。"""
    fake_store["step4_path"] = work_dir / "step4_chunks.json"
    monkeypatch.setattr(entrance, "make_embedder", lambda: _FakeEmbedder())

    entrance.run_vector(_write_source(work_dir))

    chunks = json.loads((work_dir / "step4_chunks.json").read_text(encoding="utf-8"))
    assert [chunk["id"] for chunk in chunks] == fake_store["ids"]
    assert fake_store["source"] == "样本"


def test_入口拦住与期望维数不符的embedder(work_dir, monkeypatch, fake_store):
    """conf 登记的期望维数对不上就报错：别把错维度的表建出来，建出来只能删库重入（P1·11）。"""
    fake_store["step4_path"] = work_dir / "step4_chunks.json"
    monkeypatch.setattr(entrance, "make_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(entrance, "EMBED_DIM_BY_NAME", {"fake": 4})

    with pytest.raises(ValueError) as excinfo:
        entrance.run_vector(_write_source(work_dir))

    assert "期望 4 维" in str(excinfo.value)
