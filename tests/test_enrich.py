# -*- coding: utf-8 -*-
"""
入库标签抽取的解析与「抽空看得见」：锁定 P1·12 的两半。

职责：锁定 offline/extract/enrich.py 的三件事——
      1. 抠 JSON 用花括号配对，不贪婪：模型输出「思考 + JSON」「两个 JSON 对象」都要能拿到 tags；
      2. 认不出来返回 []（不抛），来源身份 / 空项 / 重复项 / 非字符串字段都不许把入库打崩；
      3. 抽空**不是静默降级**：解析失败要打出原因与模型原始输出，入库汇总要把这些来源单独列出来。
      没有 1 会退回改前那个贪婪正则：`re.search(r"\\{.*\\}", DOTALL)` 在两个对象上从第一个 `{`
      一直跨到最后一个 `}`，json.loads 必失败 → 整个 tags 变 []，而入库照样算成功；
      没有 3 则一份「没有标签的文档」混在「成功 N」里，没人看得出来。

应用场景：改 enrich.py 的解析、或改 scripts/ingest/ingest.py 的入库汇总之后跑
          python -m pytest tests/test_enrich.py。
"""

from __future__ import annotations

import json

from offline.extract import enrich
from scripts.ingest.ingest import report_empty_tags

# ========== 解析模型输出（P1·12 前半） ==========


def test_思考文字包着JSON也能解析():
    """模型先「思考」再给 JSON 是最常见的形态，前后文字不许影响解析。"""
    raw = '先看正文：这是一份汽车手册。\n{"tags":[{"field":"品牌","value":"比亚迪"}]}\n以上。'

    assert enrich._parse_tag_json(raw, "某手册") == [{"field": "品牌", "value": "比亚迪"}]


def test_两个JSON对象取含tags的那个():
    """贪婪正则会跨对象框成一整段（必失败）；配对扫描要能拿到第二个对象。"""
    raw = '{"note":"第一个对象不是答案"}\n{"tags":[{"field":"车型","value":"汉EV"}]}'

    assert enrich._parse_tag_json(raw, "某手册") == [{"field": "车型", "value": "汉EV"}]


def test_两个对象都含tags取第一个():
    """先出现的就是答案，后面的补充说明不覆盖它。"""
    raw = '{"tags":[{"field":"甲","value":"一"}]}\n{"tags":[{"field":"乙","value":"二"}]}'

    assert enrich._parse_tag_json(raw, "某手册") == [{"field": "甲", "value": "一"}]


def test_值里带花括号不打断配对():
    """配对数法要认字符串：值里出现 `}` 不许把对象截断。"""
    raw = '{"tags":[{"field":"备注","value":"含 } 花括号"}]}'

    assert enrich._parse_tag_json(raw, "某手册") == [{"field": "备注", "value": "含 } 花括号"}]


def test_认不出JSON时返回空而不是抛():
    """入库不能因为一次模型输出变形就整份源失败。"""
    assert enrich._parse_tag_json("这份手册我抽不出标签", "某手册") == []
    assert enrich._parse_tag_json('{"tags":[', "某手册") == []
    assert enrich._parse_tag_json("", "某手册") == []


def test_来源身份空项重复项与非字符串字段都丢掉():
    """来源身份不是标签；空值、重复对丢掉，数字字段不把解析打崩。"""
    raw = json.dumps(
        {
            "tags": [
                {"field": "source", "value": "某手册"},
                {"field": "品牌", "value": "某手册"},
                {"field": "品牌", "value": "比亚迪"},
                {"field": "品牌", "value": "比亚迪"},
                {"field": "", "value": "空字段"},
                {"field": "空值", "value": "   "},
                {"field": 7, "value": "数字字段"},
                "不是对象",
            ]
        },
        ensure_ascii=False,
    )

    assert enrich._parse_tag_json(raw, "某手册") == [
        {"field": "品牌", "value": "比亚迪"},
        {"field": "7", "value": "数字字段"},
    ]


def test_最多留八条():
    """上限是 INGEST_TAG_MAX：多出来的不许进库。"""
    raw = json.dumps(
        {"tags": [{"field": f"字段{i}", "value": f"值{i}"} for i in range(1, 12)]},
        ensure_ascii=False,
    )

    tags = enrich._parse_tag_json(raw, "某手册")

    assert len(tags) == enrich.INGEST_TAG_MAX == 8
    assert tags[0] == {"field": "字段1", "value": "值1"}


# ========== 抽空要可见（P1·12 后半） ==========


def test_解析失败时打出原因与模型原始输出(capsys, monkeypatch):
    """抽空不是静默降级：要能看出是哪一步空的、模型到底给了什么。"""
    monkeypatch.setattr(enrich, "chat_text", lambda prompt: "这份手册我抽不出标签")

    tags = enrich.infer_ingest_tags("正文片段", "某手册")
    printed = capsys.readouterr().out

    assert tags == []
    assert "解析不出 tags" in printed
    assert "这份手册我抽不出标签" in printed


def test_模型调用失败也打出来(capsys, monkeypatch):
    """调用失败与解析失败是两件事，报错要能区分。"""

    def _boom(prompt: str) -> str:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    monkeypatch.setattr(enrich, "chat_text", _boom)

    tags = enrich.infer_ingest_tags("正文片段", "某手册")
    printed = capsys.readouterr().out

    assert tags == []
    assert "RuntimeError" in printed
    assert "解析不出 tags" not in printed


def test_标签抽空的来源在入库汇总里单独列出(capsys, work_dir):
    """入库汇总必须把「没有标签的来源」按名字列出来，否则它与「成功」不可区分。"""
    with_tags = work_dir / "有标签的手册" / "chunkable_source.json"
    without_tags = work_dir / "没有标签的手册" / "chunkable_source.json"
    for path, tags in (
        (with_tags, [{"field": "品牌", "value": "比亚迪"}]),
        (without_tags, []),
    ):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"source": path.parent.name, "tags": tags}, ensure_ascii=False),
            encoding="utf-8",
        )

    report_empty_tags([with_tags, without_tags])
    printed = capsys.readouterr().out

    assert printed.strip() == "标签抽空的来源 1 个（检索时不会按标签加权）：没有标签的手册"
