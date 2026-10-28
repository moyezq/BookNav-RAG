# -*- coding: utf-8 -*-
"""
embedding HTTP 客户端的分批约定：整个库不许塞进一个请求。

职责：锁定 backend/offline/vector/store.py::HttpEmbedder.embed_texts——
      1. 按 request_batch 切多次 POST，顺序拼回，结果顺序与输入一致；
      2. 段数不超过一批时只发一次；
      3. request_batch=0 才退回「整批一个请求」；
      4. 空输入不发请求；每一批都带上读超时。
      为什么要有这条约定：bge-m3 在 CPU 上实测 4.6～6.0 s/段，138 段塞一个请求是
      631～833s，超过客户端读超时后整批失败、服务端却照算完——白等一轮还占着编码锁。
      不碰网络：把 store.requests.post 换成记录用的替身。

应用场景：改 embed 客户端或入库批大小之后跑 python -m pytest tests/test_embed_client.py。
"""

from __future__ import annotations

import pytest

from backend.offline.vector.store import HttpEmbedder

URL = "http://127.0.0.1:8030/v1/embeddings"


class _FakeResponse:
    """只实现客户端读的那两个入口；把文本序号编进向量，用来验证顺序。"""

    def __init__(self, texts: list[str]):
        self._texts = texts

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        rows = []
        for index, text in enumerate(self._texts):
            suffix = text[1:]
            # "t7" 用序号，其它文本用本次批内下标——两种情况都不出问题
            rows.append({"embedding": [float(suffix) if suffix.isdigit() else float(index), 0.0]})
        return {"data": rows}


@pytest.fixture()
def posted(monkeypatch) -> list:
    """记录每次 POST 的 input 批次与 timeout。"""
    calls: list = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"input": json["input"], "timeout": timeout, "url": url})
        return _FakeResponse(json["input"])

    monkeypatch.setattr("backend.offline.vector.store.requests.post", fake_post)
    return calls


def test_分批请求后向量顺序与输入一致(posted):
    """20 段、批大小 16 → 两次 POST（16 + 4），拼回来仍是原来那 20 条且有序。"""
    embedder = HttpEmbedder(URL, "bge-m3", request_batch=16)

    vectors = embedder.embed_texts([f"t{i}" for i in range(20)])

    assert [len(call["input"]) for call in posted] == [16, 4]
    assert [vector[0] for vector in vectors] == [float(i) for i in range(20)]


def test_不超过一批时只发一次(posted):
    embedder = HttpEmbedder(URL, "bge-m3", request_batch=16)

    embedder.embed_texts(["a", "bb"])

    assert [len(call["input"]) for call in posted] == [2]


def test_批大小为0时才整批一个请求(posted):
    """0 是显式选择「不切批」，只有明确知道总量的人才该这么用。"""
    embedder = HttpEmbedder(URL, "bge-m3", request_batch=0)

    embedder.embed_texts([f"t{i}" for i in range(20)])

    assert [len(call["input"]) for call in posted] == [20]


def test_空输入不发请求(posted):
    embedder = HttpEmbedder(URL, "bge-m3", request_batch=16)

    assert embedder.embed_texts([]) == []
    assert posted == []


def test_每一批都带读超时(posted):
    """批切开了，但每一批仍要带超时，不能变成无超时请求。"""
    embedder = HttpEmbedder(URL, "bge-m3", timeout_sec=600, request_batch=16)

    embedder.embed_texts([f"t{i}" for i in range(20)])

    assert [call["timeout"] for call in posted] == [600, 600]


def test_dim从第一个向量取(posted):
    """入库要用 dim 建表列，别等到写库时才发现没取到。"""
    embedder = HttpEmbedder(URL, "bge-m3", request_batch=16)

    embedder.embed_texts(["t1", "t2"])

    assert embedder.dim == 2


def test_多批时每批报一行进度(posted, capsys):
    """整库入库在 CPU 上要十几分钟：没有进度输出，长任务会被当成卡死（2026-10-08 真发生过）。"""
    embedder = HttpEmbedder(URL, "bge-m3", request_batch=16)

    embedder.embed_texts([f"t{i}" for i in range(20)])

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    # 断言的是这行必须带的信息：第几批 / 共几批，已收几段 / 共几段
    assert "1/2" in lines[0] and "16/20" in lines[0]
    assert "2/2" in lines[1] and "20/20" in lines[1]


def test_单批不往在线日志里插话(posted, capsys):
    """在线单条查询只有一批：这条路不许新增输出。"""
    embedder = HttpEmbedder(URL, "bge-m3", request_batch=16)

    embedder.embed_texts(["后视镜怎么折叠"])

    assert capsys.readouterr().out == ""
