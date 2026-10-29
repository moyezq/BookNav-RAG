# -*- coding: utf-8 -*-
"""
在线调试题材的文件写入开关与文件名：默认关，打开时按 session 分文件。

职责：锁定三件事——
      1. backend/online/recall/retrieve.py::retrieve_hits 的 write_debug 默认取 conf 的 DEBUG_WRITE_QA，
         默认 False → 不写调试题材；
      2. backend/online/generate/answer.py::write_qa_result 在 DEBUG_WRITE_QA 关时直接返回，打开才写进磁盘；
      3. backend/online/graph.py::debug_name 组装文件名（P2·31）：带 session、客户端给的脏 session
         拼不出 qa 目录之外的路径、没 session 时退回时间戳+随机 → 并发两问产出两个文件。
      为什么要有这条：2026-10-08 实测每问覆盖写 141KB 的 step2_hits.json，
      写盘抛 PermissionError 会穿透 router 变成 HTTP 500；并发时最后写者赢，调试信息本身是坏的。
      不调 8030：假知识库 + 替身 embedder 即可跑完 recall 这一层。

应用场景：改 DEBUG_WRITE_QA、debug_name 或这两处写文件之后跑本文件。
"""

from __future__ import annotations

import inspect

from backend.conf import DEBUG_TAG_MAX_CHARS
from backend.online.generate import answer
from backend.online.graph import debug_name
from backend.online.recall import retrieve


class _FakeKnowledgeBase:
    """只要 embedder_name / dim / search 三个成员，够 retrieve_hits 跑完。"""

    embedder_name = "bge-m3"
    dim = 1024

    def search(self, query_vector, filters, top_k, query_text=None) -> list:
        return []


class _StubEmbedder:
    """不打 8030，只让 retrieve_hits 走完 search。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1024 for _ in texts]


def test_检索默认不写调试题材(work_dir, monkeypatch):
    """线上不传 write_debug 时不许碰磁盘。"""
    monkeypatch.setattr(retrieve, "QA_DIR", work_dir)
    monkeypatch.setattr(retrieve, "_build_embedder", lambda *_args, **_kwargs: _StubEmbedder())

    assert inspect.signature(retrieve.retrieve_hits).parameters["write_debug"].default is False

    hits = retrieve.retrieve_hits(_FakeKnowledgeBase(), "后视镜怎么折叠", {})

    assert hits == []
    assert not (work_dir / "step2_hits.json").exists()


def test_开关打开时写调试题材(work_dir, monkeypatch):
    """打开开关才写进磁盘：这是调试用的，不是线上路径的职责。"""
    monkeypatch.setattr(answer, "QA_DIR", work_dir)
    monkeypatch.setattr(answer, "DEBUG_WRITE_QA", True)

    answer.write_qa_result("step3_answer.json", {"question": "后视镜怎么折叠"})

    assert (work_dir / "step3_answer.json").is_file()


def test_开关关闭时不写文件(work_dir, monkeypatch):
    """关着的时候连目录都不建。"""
    monkeypatch.setattr(answer, "QA_DIR", work_dir)
    monkeypatch.setattr(answer, "DEBUG_WRITE_QA", False)

    answer.write_qa_result("step3_answer.json", {"question": "后视镜怎么折叠"})

    assert not (work_dir / "step3_answer.json").exists()


# ========== 文件名按 session 分（P2·31）：并发两问不再最后写者赢 ==========


def test_调试题材名带session():
    """文件名 = 基名 + session：同一进程里两个会话各写各的。"""
    assert debug_name("step2_hits.json", "abc-123") == "step2_hits_abc-123.json"
    assert debug_name("step3_answer.json", "abc-123") == "step3_answer_abc-123.json"
    assert debug_name("step2_hits.json", "s1") != debug_name("step2_hits.json", "s2")


def test_没有session时两次调用也不同():
    """没 session（命令行）也不能互相覆盖：退回时间戳 + 随机。"""
    first = debug_name("step2_hits.json", "")
    second = debug_name("step2_hits.json", "")

    assert first != second
    assert first.startswith("step2_hits_") and first.endswith(".json")


def test_脏session写不出qa目录():
    """session 由客户端给：路径分隔符与 `..` 不许拼进文件名。"""
    for dirty in ("../../evil", "a/b\\c", "..", ".", "会话一"):
        name = debug_name("step2_hits.json", dirty)

        assert "/" not in name and "\\" not in name
        assert name.count(".") == 1
        assert ".." not in name


def test_超长session截断到上限():
    """文件名不许被客户端拖成任意长。"""
    name = debug_name("step2_hits.json", "x" * 100)

    assert name == "step2_hits_" + "x" * DEBUG_TAG_MAX_CHARS + ".json"


def test_两个会话各写各的调试题材(work_dir, monkeypatch):
    """P2·31 的验收：同一进程两次提问产出两个文件，不是一个被覆盖的文件。"""
    monkeypatch.setattr(answer, "QA_DIR", work_dir)
    monkeypatch.setattr(answer, "DEBUG_WRITE_QA", True)

    answer.write_qa_result(debug_name("step3_answer.json", "session-a"), {"question": "甲的问题"})
    answer.write_qa_result(debug_name("step3_answer.json", "session-b"), {"question": "乙的问题"})

    written = sorted(path.name for path in work_dir.glob("step3_answer_*.json"))
    assert written == ["step3_answer_session-a.json", "step3_answer_session-b.json"]
    assert not (work_dir / "step3_answer.json").exists()


def test_召回按给定的名字写调试文件(work_dir, monkeypatch):
    """召回层也用调用方给的调试题材名，不再写死 step2_hits.json。"""
    monkeypatch.setattr(retrieve, "QA_DIR", work_dir)
    monkeypatch.setattr(retrieve, "_build_embedder", lambda *_args, **_kwargs: _StubEmbedder())

    hits = retrieve.retrieve_hits(
        _FakeKnowledgeBase(),
        "后视镜怎么折叠",
        {},
        write_debug=True,
        debug_name="step2_hits_session-a.json",
    )

    assert hits == []
    assert (work_dir / "step2_hits_session-a.json").is_file()
    assert not (work_dir / "step2_hits.json").exists()
