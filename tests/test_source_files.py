# -*- coding: utf-8 -*-
"""
源文件入口的扩展名允许清单。

职责：锁定 backend/offline/extract/entrance.py::list_source_files 的三件事——
      1. 不在允许清单里的后缀（`backend/data/source/README.txt` 这类说明文件）不进抽取，且跳过的文件名要打出来；
      2. 一个可入库的都没有 → FileNotFoundError，消息里给出支持的后缀；
      3. 后缀大小写不敏感（Windows 上 `手册.PDF` 本来就该入库）。
      为什么要有这条：加允许清单之前 README.txt 会被当成源进抽取，在路由被拒后
      scripts/ingest/ingest.py 只打印一行「失败」并且退 0，看着像入库成功。

应用场景：改 SOURCE_ALLOWED_SUFFIXES 或 list_source_files 之后跑本文件。
"""

from __future__ import annotations

import pytest

from backend.offline.extract import entrance


def _make_sources(work_dir, names: list[str]):
    """在临时目录里造同名文件，返回该目录（调用方把它当成 SOURCE_DIR）。"""
    for name in names:
        (work_dir / name).write_text("x", encoding="utf-8")
    return work_dir


def test_跳过不在允许清单里的源并报出文件名(work_dir, monkeypatch, capsys):
    """README.txt 不许进抽取，跳过要看得见，不许静默丢。"""
    _make_sources(work_dir, ["手册.pdf", "README.txt"])
    monkeypatch.setattr(entrance, "SOURCE_DIR", work_dir)

    paths = entrance.list_source_files()

    assert [path.name for path in paths] == ["手册.pdf"]
    assert "README.txt" in capsys.readouterr().out


def test_没有可入库源时报出支持的后缀(work_dir, monkeypatch):
    """全是说明文件时不能静默返回空列表：报错要说清支持什么。"""
    _make_sources(work_dir, ["README.txt"])
    monkeypatch.setattr(entrance, "SOURCE_DIR", work_dir)

    with pytest.raises(FileNotFoundError) as excinfo:
        entrance.list_source_files()

    assert ".pdf" in str(excinfo.value)


def test_后缀大小写不敏感(work_dir, monkeypatch):
    """`手册.PDF` 与 `手册.pdf` 是同一类源。"""
    _make_sources(work_dir, ["手册.PDF"])
    monkeypatch.setattr(entrance, "SOURCE_DIR", work_dir)

    assert [path.name for path in entrance.list_source_files()] == ["手册.PDF"]
