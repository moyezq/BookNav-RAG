# -*- coding: utf-8 -*-
"""
分块与页码的单元测试。

职责：锁定 offline/vector/chunk.py 里三件必须成立的事——
      1. 超限切出来的每个子块只带自己覆盖的页（P0·5；以前是整节页码复制给每个子块）；
      2. 页归属：一页只归第一条命中的目录区间，导航页不进；
      3. token 估算的当前做法（当前做法若改，这里必须同轮改）。

应用场景：改 chunk.py 之后跑 python -m pytest tests/test_chunk.py。
"""

from __future__ import annotations

import pytest

from offline.vector.chunk import (
    _assemble_pieces_with_pages,
    assign_content_to_catalogue,
    chunks_from_assigned_rows,
    clean_page_text,
    estimate_tokens_by_chars,
    is_navigation_page,
    navigation_page_reason,
)

# ========== 页码下沉到块级（P0·5） ==========


def test_超限切块后页码跟着块走():
    """一节切成多块时，每块只带自己覆盖的页，不带整节页。"""
    # 每页正文约 1000 字 → 估算 500 token；上限 600 → 每块只装得下一页
    texts = ["甲" * 1000, "乙" * 1000, "丙" * 1000]
    pages = [10, 11, 12]
    chunks = chunks_from_assigned_rows(
        [
            {"page": 10, "catalogue_title": "某节", "text": texts[0]},
            {"page": 11, "catalogue_title": "某节", "text": texts[1]},
            {"page": 12, "catalogue_title": "某节", "text": texts[2]},
        ],
        "样本",
        [],
        "",
        max_doc_tokens=600,
    )

    assert len(chunks) == 3
    assert [chunk["metadata"]["pages"] for chunk in chunks] == [[10], [11], [12]]
    assert [chunk["metadata"]["part"] for chunk in chunks] == [0, 1, 2]
    # 声明页次合计 == 真实页次合计；这是页精度的定义
    declared = sum(len(chunk["metadata"]["pages"]) for chunk in chunks)
    assert declared == len(pages)


def test_一节装得下时页码就是整节():
    """不超限时不切，页码就是该节的页。"""
    rows = [
        {"page": 1, "catalogue_title": "前言", "text": "短正文"},
        {"page": 2, "catalogue_title": "前言", "text": "短正文二"},
    ]
    chunks = chunks_from_assigned_rows(rows, "样本", [], "", max_doc_tokens=1800)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["pages"] == [1, 2]
    assert chunks[0]["metadata"]["part"] == 0


@pytest.mark.xfail(
    strict=True,
    reason="P2·14：单段超限无兜底——split_doc_text_into_chunks 只按空行分段，整页无空行时原样成块",
)
def test_单页自身超限时只切文本不伪造页码():
    """期望行为：一页超过上限时按段再切，切出的每段都算这一页，且不引入别的页号。

    现状（P2·14 未修）：整页无空行 → 只有一个段落 → 不切，仍是 1 块。
    修完 P2·14 后本用例会 XPASS，strict 会把它变成失败，届时删掉本标记。
    """
    long_page = "甲" * 3000
    rows = [{"page": 7, "catalogue_title": "某节", "text": long_page}]
    chunks = chunks_from_assigned_rows(rows, "样本", [], "", max_doc_tokens=600)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["metadata"]["pages"] == [7]
        assert chunk["metadata"]["part"] >= 0


def test_空正文的页不会被算进任何块():
    """页码只跟着真正有正文的页走。"""
    rows = [
        {"page": 1, "catalogue_title": "前言", "text": "正文"},
        {"page": 2, "catalogue_title": "前言", "text": ""},
    ]
    chunks = chunks_from_assigned_rows(rows, "样本", [], "", max_doc_tokens=1800)

    assert chunks[0]["metadata"]["pages"] == [1]


def test_装配函数返回的块正文不含分隔符():
    """页间分隔用的分隔符必须还原成空行，不能漏进正文。"""
    packed = _assemble_pieces_with_pages(["第一页" * 100, "第二页" * 100], [1, 2], 100000)

    assert len(packed) == 1
    text, pages = packed[0]
    assert "\x00" not in text
    assert pages == [1, 2]
    assert "\n\n" in text


# ========== 段边界：分段累积分块的前提（P0·6） ==========


def test_清洗保留段边界():
    """空行是段边界，必须留着；段内换行才压成空格。分段累积分块靠的就是它。"""
    cleaned = clean_page_text("第一段第一行\n第一段第二行\n\n第二段")

    assert cleaned == "第一段第一行第一段第二行\n\n第二段"


def test_清洗去掉井号图片但仍分段():
    """段内清洗照旧（井号、图片），段边界不许被一起抹掉。"""
    cleaned = clean_page_text("#### 小标题\n正文\n\n![图](a.png)\n第二段")

    assert cleaned == "小标题正文\n\n第二段"


def test_单页超限时按空行分段切开():
    """清洗后仍带段边界，所以一页超限会**按段**切开，而不是原样成一个超限块（P0·6）。"""
    page = clean_page_text("\n\n".join(["甲" * 800, "乙" * 800, "丙" * 800]))
    rows = [{"page": 7, "catalogue_title": "某节", "text": page}]

    chunks = chunks_from_assigned_rows(rows, "样本", [], "", max_doc_tokens=600)

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk["metadata"]["pages"] == [7]
        assert chunk["metadata"]["part"] >= 0


# ========== 页归属 ==========


def test_一页只归第一条命中的区间():
    """区间用半开 [start, end)；重叠时取先命中的那条。"""
    pages = [{"page": 5, "text": "这里是一段足够长的正文内容用来归类"}]
    ranges = [
        {"title": "甲节", "start_page": 1, "end_page": 10},
        {"title": "乙节", "start_page": 5, "end_page": 20},
    ]
    rows = assign_content_to_catalogue(pages, ranges)

    assert len(rows) == 1
    assert rows[0]["catalogue_title"] == "甲节"


def test_所有区间外的页会被丢弃且不报错():
    """按目录分页这一步确实不说话（P1·8）：丢页由 scripts/ingest/verify_ingest.py 分类后报出来。"""
    pages = [{"page": 99, "text": "区间外的正文"}]
    ranges = [{"title": "甲节", "start_page": 1, "end_page": 10}]

    assert assign_content_to_catalogue(pages, ranges) == []


def test_导航页不进正文():
    """目录页、图片索引、章扉页不算正文。"""
    assert is_navigation_page("目 录\n第一章 ........ 1\n第二章 ........ 10") is True
    assert is_navigation_page("#### 正常小节\n操作步骤见下。") is False


def test_导航页判定能说出原因():
    """分类靠 navigation_page_reason：目录 / 索引 / 章扉页各归各的，正文返回 None。"""
    chapter_cover = "3\n" + "\n".join(f"小节{i}......... {i}" for i in range(1, 7))

    assert navigation_page_reason("目 录\n第一章 ........ 1") == "目录"
    assert navigation_page_reason("图片索引\n|插图|说明|") == "索引"
    assert navigation_page_reason("字母索引 387\nA 项目 ........ 12") == "索引"
    assert navigation_page_reason(chapter_cover) == "章扉页"
    assert navigation_page_reason("#### 正常小节\n操作步骤见下。") is None


def test_导航页判定的布尔包装与原因判定一致():
    """is_navigation_page 只是 navigation_page_reason 的布尔包装（只看有没有原因），两者不许分叉。"""
    samples = [
        "目 录\n第一章 ........ 1",
        "图片索引\n|插图|说明|",
        "3\n" + "\n".join(f"小节{i}......... {i}" for i in range(1, 7)),
        "#### 正常小节\n操作步骤见下。",
        "",
    ]

    for text in samples:
        assert is_navigation_page(text) is (navigation_page_reason(text) is not None)


# ========== token 估算的当前做法 ==========


def test_token估算按字符数一半():
    """当前做法：中文约 2 字 1 token。这个做法改了，本用例要同轮改。"""
    assert estimate_tokens_by_chars("") == 1
    assert estimate_tokens_by_chars("一二三四") == 2
    assert estimate_tokens_by_chars("a" * 101) == 50
