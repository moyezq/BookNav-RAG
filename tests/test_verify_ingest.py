# -*- coding: utf-8 -*-
"""
入库验收脚本自身的测试：页精度与丢页分类判据的判别力与误报边界。

职责：锁定 scripts/ingest/verify_ingest.py::check_ingest 的页精度判据——
      1. 同一节的子块各自只覆盖一部分页（块级页码）→ 放行；
      2. 相邻 part 都声明整节页、重叠 ≥2 页（块级页码之前的样子，实测 3.21 倍）→ 报硬问题；
      3. 单页超限被按段切开，同一页出现在相邻两块 → **不报**硬问题，只进「跨块共用的页」指标；
      4. 同一页落在两个目录节 → 报硬问题；
      5. 块 id 重复 → 报硬问题；
      6. step4 与 backend/data/lancedb 的块 id / 页码不一致 → 报硬问题（半成品入库）。

      丢页一端：
      7. 导航页（目录 / 索引 / 章扉页）与空页按设计丢 → 分类报出并放行；
      8. 有正文却没按目录分页（区间内未分到页 / 首条目录项之前）→ 报硬问题，阈值可调；
      9. 判成章扉页却留着大段正文 → 只记「疑似误杀」，不进硬断言（启发式）。
      没有 7 就退回「N 页有正文，需逐个确认」；没有 8 会让丢正文静默；没有 9 会让近似规则
      吃掉正文页而无声。没有 2 会漏掉真回归；没有 3 会变成常驻红线（正确数据也判未过）；
      没有 6 会漏掉「step4 已经是新的、在线读的库还是旧的」这种两面不一致。

应用场景：改 verify_ingest.py 的页精度或丢页分类判据之后跑 python -m pytest tests/test_verify_ingest.py。
"""

from __future__ import annotations

from scripts.ingest.verify_ingest import check_ingest, check_store_sync

SOURCE = {
    "source": "样本",
    "pages": [{"page": 1, "text": "正文一"}, {"page": 2, "text": "正文二"}],
    "catalogue_ranges": [],
    "tags": [],
}


def _chunk(chunk_id: str, text: str, pages: list, part: int, title: str = "某节") -> dict:
    """造一个知识块：页精度只看 metadata.pages、part 与 catalogue_title。"""
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source": "样本",
            "tags": [],
            "knowledge_type": "",
            "catalogue_title": title,
            "pages": pages,
            "part": part,
        },
    }


def test_块级页码不重叠时验收放行():
    """块级页码：part 0 带第 1 页、part 1 带第 2 页，无重叠，无硬问题。"""
    chunks = [_chunk("样本-0001", "正文一", [1], 0), _chunk("样本-0002", "正文二", [2], 1)]

    result = check_ingest(SOURCE, chunks)

    assert result["page_precision"] == 1.0
    assert result["duplicated_pages"] == []
    assert result["overlapped_pairs"] == 0
    assert result["hard_problems"] == []


def test_整节页码被复制时硬断言报出来():
    """块级页码之前：两个 part 都声明 [1, 2]，重叠 2 页 → 必须进硬问题。"""
    chunks = [_chunk("样本-0001", "正文一", [1, 2], 0), _chunk("样本-0002", "正文二", [1, 2], 1)]

    result = check_ingest(SOURCE, chunks)

    assert result["page_precision"] == 2.0
    assert result["overlapped_pairs"] == 1
    assert result["overlapped_sections"] == ["某节"]
    assert result["hard_problems"], "整节页码复制必须让验收脚本非零退出"


def test_单页被按段切开跨两块不算硬问题():
    """一页超限按段切开后，切分处那一页会出现在相邻两块里——正当，不许报硬问题。"""
    chunks = [
        _chunk("样本-0001", "正文一前半", [1], 0),
        _chunk("样本-0002", "正文一后半与正文二", [1, 2], 1),
    ]

    result = check_ingest(SOURCE, chunks)

    assert result["duplicated_pages"] == [1]  # 指标如实报出跨块共用
    assert result["overlapped_pairs"] == 0  # 但相邻 part 只共享 1 页，不是回归
    assert result["hard_problems"] == []


def test_同一页落在两个目录节是硬问题():
    """一页只归第一条命中的区间；跨节重复说明归错了。"""
    chunks = [
        _chunk("样本-0001", "正文一", [1], 0, title="甲节"),
        _chunk("样本-0002", "正文一又一遍", [1], 0, title="乙节"),
    ]

    result = check_ingest(SOURCE, chunks)

    assert result["cross_section"] == [1]
    assert result["hard_problems"]


def test_块id重复也算硬问题():
    """id 唯一是向量层按 source 整批替换后还能定位到块的前提。"""
    chunks = [_chunk("样本-0001", "正文一", [1], 0), _chunk("样本-0001", "正文二", [2], 1)]

    result = check_ingest(SOURCE, chunks)

    assert result["ids_duplicated"] == ["样本-0001"]
    assert result["hard_problems"]


# ========== 向量库与 step4 同步（跑在线读的是库） ==========


def _record(chunk_id: str, pages: list, source: str = "样本") -> dict:
    """造一条向量库记录：同步只看 id / source / pages。"""
    return {"id": chunk_id, "source": source, "pages": pages, "text": "正文", "metadata": {"pages": pages}}


def test_库与step4一致时放行():
    chunks = [_chunk("样本-0001", "正文一", [1], 0), _chunk("样本-0002", "正文二", [2], 1)]
    records = [_record("样本-0001", [1]), _record("样本-0002", [2])]

    result = check_store_sync("样本", chunks, records)

    assert result["store_rows"] == 2
    assert result["step4_rows"] == 2
    assert result["ids_only_in_step4"] == []
    assert result["ids_only_in_store"] == []
    assert result["page_mismatches"] == []
    assert result["hard_problems"] == []


def test_库里缺块是硬问题():
    """embed 失败时 step4 已经写进磁盘而库没写：在线检索读库，答案会少一批。"""
    chunks = [_chunk("样本-0001", "正文一", [1], 0), _chunk("样本-0002", "正文二", [2], 1)]
    records = [_record("样本-0001", [1])]

    result = check_store_sync("样本", chunks, records)

    assert result["ids_only_in_step4"] == ["样本-0002"]
    assert result["hard_problems"]


def test_页码不同是硬问题():
    """分块改了页码而库还是旧的：引用页号会指错。"""
    chunks = [_chunk("样本-0001", "正文一", [1, 2], 0)]
    records = [_record("样本-0001", [1])]

    result = check_store_sync("样本", chunks, records)

    assert result["page_mismatches"] == ["样本-0001"]
    assert result["hard_problems"]


def test_库里有step4没有的旧块是硬问题():
    """按来源整批替换没生效：旧块留着照样会被检索到。"""
    chunks = [_chunk("样本-0001", "正文一", [1], 0)]
    records = [_record("样本-0001", [1]), _record("样本-9999", [9])]

    result = check_store_sync("样本", chunks, records)

    assert result["ids_only_in_store"] == ["样本-9999"]
    assert result["hard_problems"]


def test_别的来源的块不参与比对():
    """chunks 表里还躺着别的来源，不该把它算成本来源的差异。"""
    chunks = [_chunk("样本-0001", "正文一", [1], 0)]
    records = [_record("样本-0001", [1]), _record("别的-0001", [7], source="别的")]

    result = check_store_sync("样本", chunks, records)

    assert result["store_rows"] == 1
    assert result["hard_problems"] == []


# ========== 丢页原因分类与阈值告警 ==========


def _source(pages: list, ranges: list) -> dict:
    """造一份可分块原文：丢页分类只看 pages、catalogue_ranges 与页文本。"""
    return {"source": "样本", "pages": pages, "catalogue_ranges": ranges, "tags": []}


def test_导航页与空页丢页按原因分类且放行():
    """目录 / 图片索引 / 章扉页 / 空页都是按设计丢的：原因要分开报，但不算硬问题。"""
    source = _source(
        [
            {"page": 1, "text": "目 录\n第一章 ........ 1\n第二章 ........ 10"},
            {"page": 2, "text": "图片索引\n|插图|说明|\n|---|---|\n|1 安全|通读本部分|"},
            {
                "page": 3,
                "text": (
                    "3\n\n# 第一章 1\n\n"
                    "|**1-1**小节一**........14**|\n"
                    "|小节二.........15|\n"
                    "|小节三.........16|\n"
                    "|小节四.........17|\n"
                    "|小节五.........18|\n"
                ),
            },
            {"page": 4, "text": "   "},
            {"page": 5, "text": "正文五"},
        ],
        [{"title": "第一章", "start_page": 1, "end_page": 10**9}],
    )

    result = check_ingest(source, [_chunk("样本-0001", "正文五", [5], 0)])

    assert result["missing_pages"] == [1, 2, 3, 4]
    assert result["missing_classification"]["counts"] == {
        "目录": 1,
        "索引": 1,
        "章扉页": 1,
        "空文本": 1,
    }
    assert result["missing_classification"]["content_pages"] == []
    assert result["hard_problems"] == []


def test_有正文却没分到页超过阈值就报硬问题():
    """目录区间覆盖了这页，它却一个字都没进块——这就是要拦的静默丢正文。"""
    source = _source(
        [{"page": 1, "text": "正文一"}, {"page": 2, "text": "正文二"}],
        [{"title": "某节", "start_page": 1, "end_page": 10**9}],
    )

    result = check_ingest(source, [_chunk("样本-0001", "正文一", [1], 0)])

    assert result["missing_classification"]["reasons"] == {2: "区间内未分到页"}
    assert result["missing_classification"]["content_pages"] == [2]
    assert result["hard_problems"], "有正文的页被丢掉必须让验收脚本非零退出"


def test_丢页阈值是容忍页数不是写死零():
    """留 1 页额度时，同一份产出放行——阈值可调，不是常驻红线。"""
    source = _source(
        [{"page": 1, "text": "正文一"}, {"page": 2, "text": "正文二"}],
        [{"title": "某节", "start_page": 1, "end_page": 10**9}],
    )

    result = check_ingest(source, [_chunk("样本-0001", "正文一", [1], 0)], missing_content_max=1)

    assert result["missing_classification"]["content_pages"] == [2]
    assert result["hard_problems"] == []


def test_首条目录项之前的丢页也按真丢正文算():
    """封面 / 前言在目录第一条之前：目录不收录它们，但正文丢了必须看得见。"""
    source = _source(
        [{"page": 1, "text": "封面标题"}, {"page": 2, "text": "正文二"}],
        [{"title": "某节", "start_page": 2, "end_page": 10**9}],
    )

    result = check_ingest(source, [_chunk("样本-0001", "正文二", [2], 0)])

    assert result["missing_classification"]["reasons"] == {1: "首条目录项之前"}
    assert result["hard_problems"]


def test_判成章扉页却留着大段正文记疑似误杀():
    """「点线行 ≥5 且过半」是近似规则：正文页列了 5 条以上编号项就会被整页吃掉。"""
    source = _source(
        [
            {
                "page": 1,
                "text": (
                    "4\n\n# 第二章\n\n"
                    "|**2-1**小节一**........20**|\n"
                    "|小节二.........21|\n"
                    "|小节三.........22|\n"
                    "|小节四.........23|\n"
                    "|小节五.........24|\n\n"
                    + "这一页是真正的正文，" * 25
                ),
            },
            {"page": 2, "text": "正文二"},
        ],
        [{"title": "某节", "start_page": 1, "end_page": 10**9}],
    )
    chunks = [_chunk("样本-0001", "正文二", [2], 0)]

    result = check_ingest(source, chunks)

    assert result["missing_classification"]["reasons"] == {1: "章扉页"}
    assert result["missing_classification"]["suspect_navigation_pages"] == [1]
    assert result["problems"], "疑似误杀要说出来"
    assert result["hard_problems"] == [], "启发式不进硬断言"

    # 边界：阈值抬高到残留正文之上就不再怀疑
    loose = check_ingest(source, chunks, suspect_body_chars=1000)
    assert loose["missing_classification"]["suspect_navigation_pages"] == []


def test_只剩图片的页算清洗后为空():
    """有文本但清洗后什么都不剩：算空页，不算丢正文。"""
    source = _source(
        [{"page": 1, "text": "![](某图.png)"}, {"page": 2, "text": "正文二"}],
        [{"title": "某节", "start_page": 1, "end_page": 10**9}],
    )

    result = check_ingest(source, [_chunk("样本-0001", "正文二", [2], 0)])

    assert result["missing_classification"]["reasons"] == {1: "清洗后为空"}
    assert result["hard_problems"] == []


def test_没有目录区间的来源丢页单列不报硬问题():
    """无目录时 pages 恒为 []：整份来源的页都会「丢」，那是已知缺口不是新事故。"""
    source = _source([{"page": 1, "text": "正文一"}, {"page": 2, "text": "正文二"}], [])

    result = check_ingest(source, [])

    assert set(result["missing_classification"]["reasons"].values()) == {"来源没有目录区间"}
    assert result["hard_problems"] == []
    assert result["problems"], "没有目录区间要说出来，不能静默"
