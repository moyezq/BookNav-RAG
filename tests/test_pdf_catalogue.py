# -*- coding: utf-8 -*-
"""
印刷目录页的挑选：目录页数不固定，不许写死「只读前 15 页」。

职责：锁定 backend/offline/extract/catalogue.py::_select_catalogue_pages 的四件事——
      1. 目录超过 15 页时全部挑进来（写死 15 页会把后半本书并进倒数第二个标题）；
      2. 目录之后的正文页不会被当成目录页；
      3. 续页不重复「目 录」表头也算目录页（紧接上一页目录页即可）；
      4. 中间隔着图片索引时，后面的章节小目录不会被吸进来（真实手册第 4～6 页是目录、
         第 7～12 页是图片索引、第 13 页是带点线的章节小目录）。
      为什么要有这条：2026-10-08 实测那份 390 页手册的目录只落在第 4～6 页，
      写死 15 页的后果在目录页数更多的文档上才会现形——那时区间会静默错，不会报错。

应用场景：改 _select_catalogue_pages 或 conf 的目录扫描常量之后跑本文件。
"""

from __future__ import annotations

from backend.offline.extract.catalogue import (
    _build_catalogue_ranges,
    _extract_catalogue_lines,
    _select_catalogue_pages,
    _split_joined_toc_cell,
)


def _toc_page(page_no: int, *, header: bool = True, entries: int = 3) -> dict:
    """造一页目录：表头可选，每条 `标题........页码` 都带点线。"""
    lines = ["目 录"] if header else []
    for index in range(entries):
        lines.append(f"章节{page_no}之{index}........{page_no * 3 + index}")
    return {"page": page_no, "text": "\n".join(lines)}


def _body_page(page_no: int) -> dict:
    """普通正文页：没有点线目录行。"""
    return {"page": page_no, "text": "研究表明，正确使用安全带能显著降低碰撞伤害。"}


def _dotted_section_page(page_no: int, entries: int = 5) -> dict:
    """章节小目录页：带点线，但不是整本书的印刷目录（真实手册第 13 页）。"""
    lines = [f"> 安全 {page_no}"]
    for index in range(entries):
        lines.append(f"1-{index} 某节........{page_no + index}")
    return {"page": page_no, "text": "\n".join(lines)}


def test_目录超过十五页时全部挑进来():
    """20 页目录必须整段挑到；挑漏一页，那一页的条目就落进上一节区间。"""
    pages = [_toc_page(page_no) for page_no in range(1, 21)]

    selected = _select_catalogue_pages(pages)

    assert [page["page"] for page in selected] == list(range(1, 21))
    ranges = _build_catalogue_ranges(_extract_catalogue_lines(selected))
    assert len(ranges) == 20 * 3
    assert ranges[-1]["start_page"] == 20 * 3 + 2


def test_目录之后的正文页不算目录页():
    """正文页没有点线目录行，扫到就该停，不许把整本书当目录。"""
    pages = [_toc_page(page_no) for page_no in range(1, 4)]
    pages.extend(_body_page(page_no) for page_no in range(4, 20))

    selected = _select_catalogue_pages(pages)

    assert [page["page"] for page in selected] == [1, 2, 3]


def test_续页没有表头也算目录页():
    """续页不重复「目 录」是常见排版；紧接上一页目录页且能解析出目录行即可。"""
    pages = [_toc_page(1), _toc_page(2, header=False), _body_page(3)]

    selected = _select_catalogue_pages(pages)

    assert [page["page"] for page in selected] == [1, 2]


def test_隔着图片索引的章节小目录不被吸进来():
    """图片索引页没有目录行 → 连续三页不是目录页就停，第 7 页的章节小目录不许进。"""
    pages = [_toc_page(page_no) for page_no in range(1, 4)]
    pages.extend(
        {"page": page_no, "text": f"图片索引 {page_no}\n![](某图.png)"}
        for page_no in range(4, 7)
    )
    pages.append(_dotted_section_page(7))

    selected = _select_catalogue_pages(pages)

    assert [page["page"] for page in selected] == [1, 2, 3]


def test_没有目录页时返回空列表():
    """一份没有印刷目录的源：挑不出目录页，交给上层走「无目录」那一支。"""
    pages = [_body_page(page_no) for page_no in range(1, 6)]

    assert _select_catalogue_pages(pages) == []


def test_同一行被拼成多条点线目录时能拆开():
    """pymupdf4llm 偶发把多条目录收成一行；不拆开会合成一条错误标题。"""
    joined = "安全须知........2 折叠支架........3 展开支架........4"
    assert _split_joined_toc_cell(joined) == [
        "安全须知........2",
        "折叠支架........3",
        "展开支架........4",
    ]
    assert _split_joined_toc_cell("单独一条........9") == ["单独一条........9"]

    pages = [
        {
            "page": 1,
            "text": "目 录\n\n安全须知........2 折叠支架........3 充电说明........5\n",
        }
    ]
    lines = _extract_catalogue_lines(pages)
    assert lines == ["安全须知 2", "折叠支架 3", "充电说明 5"]
    ranges = _build_catalogue_ranges(lines)
    assert [item["title"] for item in ranges] == ["安全须知", "折叠支架", "充电说明"]
    assert [item["start_page"] for item in ranges] == [2, 3, 5]
