# -*- coding: utf-8 -*-
"""
评测样本与评测口的测试：样本文件自洽，拒答阈值扫点真有分辨力。

职责：锁定四件事——
      1. scripts/eval/eval_samples.json 的形状与自洽：id / question 唯一、条数 ≥50、分组认得、
         hit 样本必带 source 与 pages、负与闲聊样本不带 source/pages、requires_sources 只引用
         样本文件自己声明的来源；
      2. 多轮样本的 turns 形状与图写进 state 的轮次一致（question / text / answer / route /
         filters / tags）——少 answer 键会让 backend.online.intent.classify.format_turns 抛 KeyError，
         评测会在中途失败而不是给出结论（本条是实测踩到过的注意点）；
      3. 命中判据按来源与页：来源对但页不相交不算中、页相交但来源不对也不算中；
      4. calibrate_threshold 的扫点真有分辨力：负样本余弦全在正样本之下时必须报出
         「不漏拒的最高阈值」且零误拒；负样本余弦高于正样本时不许假装有解（best_recall=None）。
      没有 4 会得到一个「扫出来但没分辨力」的校准口。

应用场景：改 eval_samples.json 或 eval_kb.py 的判定之后跑 python -m pytest tests/test_eval_samples.py。
"""

from __future__ import annotations

import json

import pytest

from backend.online.intent.classify import format_turns
from scripts.eval.eval_kb import SAMPLES_PATH, _hit_at, _pages_overlap, calibrate_threshold, load_samples

# 样本文件里允许出现的分组；新加分组要连测试一起改，避免拼错组名还静默算数
GROUPS = {
    "pos-single-doc",
    "pos-paraphrase",
    "pos-source-filter",
    "pos-identity-followup",
    "pos-cross-source",  # 仅本机 private 大样本集使用
    "chitchat",
    "neg-outdomain",
    "neg-near-domain",
}
# 图写进 state 的轮次字段（backend/online/graph.py 的 generate 节点）；format_turns 会读 question 与 answer
TURN_KEYS = {"question", "text", "answer", "route", "filters", "tags"}


def _samples() -> list[dict]:
    """读真实样本文件；解析失败就让用例失败，不要静默跳过。"""
    return load_samples(SAMPLES_PATH)["samples"]


def test_样本文件形状与自洽():
    """样本数、唯一性、分组、按 expect 的必填字段。"""
    payload = load_samples(SAMPLES_PATH)
    samples = payload["samples"]
    # 公开仓默认是 demo_handbook 小样本集；本机可换 eval_samples.private.json（≥50）
    assert len(samples) >= 8, f"评测集至少要 8 条，现在 {len(samples)} 条"
    assert payload["corpus_sources"], "corpus_sources 不能为空"

    ids = [sample["id"] for sample in samples]
    questions = [sample["question"] for sample in samples]
    assert len(ids) == len(set(ids)), "样本 id 有重复"
    assert len(questions) == len(set(questions)), "样本问句有重复"

    unknown = {sample["group"] for sample in samples} - GROUPS
    assert not unknown, f"有认不出的分组：{sorted(unknown)}"

    for sample in samples:
        if sample["expect"] == "hit":
            assert sample.get("source"), f"{sample['id']} 是 hit 却没有 source"
            assert sample.get("pages"), f"{sample['id']} 是 hit 却没有 pages"
            assert sample["source"] in payload["corpus_sources"], f"{sample['id']} 的来源不在 corpus_sources"
        else:
            assert "source" not in sample, f"{sample['id']} 不是 hit 却带 source"
            assert "pages" not in sample, f"{sample['id']} 不是 hit 却带 pages"
        for name in sample.get("requires_sources", []):
            assert name in payload["corpus_sources"], f"{sample['id']} 的 requires_sources 不在 corpus_sources"
        if sample.get("filters"):
            assert sample["filters"].get("source"), f"{sample['id']} 的 filters 没有 source"


def test_多轮样本的轮次形状与图一致():
    """turns 必须能直接喂给 format_turns：字段齐、内容非空。"""
    multi = [sample for sample in _samples() if sample.get("turns")]
    assert multi, "样本文件里至少要有一条多轮样本（身份追问）"
    for sample in multi:
        for turn in sample["turns"]:
            missing = TURN_KEYS - set(turn)
            assert not missing, f"{sample['id']} 的轮次缺字段 {sorted(missing)}"
            assert turn["question"].strip(), f"{sample['id']} 的轮次缺问句"
            assert turn["answer"].strip(), f"{sample['id']} 的轮次缺答案"
        history = format_turns(sample["turns"])
        assert "助手：" in history, f"{sample['id']} 的历史拼不出助手行"


def test_命中判据按来源与页():
    """来源与页两维都要对：任一维不对就不算中。"""
    hits = [
        {"source": "甲", "pages": [1, 2]},
        {"source": "乙", "pages": [9]},
    ]
    assert _hit_at(hits, 5, "乙", [9]) is True
    assert _hit_at(hits, 5, "乙", [8]) is False, "页不相交不该算中"
    assert _hit_at(hits, 5, "丙", [9]) is False, "来源不符不该算中"
    assert _hit_at(hits, 1, "乙", [9]) is False, "第 2 位不该算进 Hit@1"
    assert _pages_overlap([1, 2], [2, 3]) is True
    assert _pages_overlap([], [2]) is False


def _row(sample_id: str, expect: str, cosine: float) -> dict:
    """造一条已在库里跑过的样本结果：只保留扫点要读的字段。"""
    return {
        "id": sample_id,
        "group": "neg-outdomain" if expect == "refuse" else "pos-single-doc",
        "question": sample_id,
        "expect": expect,
        "ok": None,
        "skipped": False,
        "reason": "",
        "hits": [{"vector_score": cosine, "text": sample_id, "pages": [1], "source": "样本", "score": cosine}],
        "rewritten": sample_id,
        "route": "retrieve",
        "vector_score": cosine,
    }


def test_阈值扫点能分辨正负样本():
    """负样本余弦 0.30、正样本 0.90 / 0.75：分界必须停在两者之间，且边界不是空的。"""
    rows = [_row("pos-1", "hit", 0.90), _row("pos-2", "hit", 0.75), _row("neg-1", "refuse", 0.30)]
    result = calibrate_threshold(rows)
    assert result["positives"] == 2 and result["negatives"] == 1
    assert result["rows"], "扫点结果不该为空"
    best = result["best_recall"]
    assert best is not None, "负样本全在正样本之下，必须存在零漏拒的阈值"
    assert best["false_negative"] == 0 and best["false_positive"] == 0
    assert best["precision"] == 1.0 and best["recall"] == 1.0
    assert 0.30 < best["threshold"] <= 0.75, f"分界该落在两条样本之间：{best['threshold']}"
    # 边界不是空的：再高 0.01 就会把最弱的正样本判成拒答
    above = [item for item in result["rows"] if abs(item["threshold"] - 0.76) < 1e-9][0]
    assert above["false_positive"] == 1, "越过最弱正样本之后必须出现误拒"


def test_阈值扫点认得出无解():
    """负样本余弦高于正样本：任何阈值都会漏拒，不许假装有解。"""
    rows = [_row("pos-1", "hit", 0.80), _row("neg-1", "refuse", 0.95)]
    result = calibrate_threshold(rows)
    assert result["rows"]
    assert result["best_recall"] is None, "负样本比正样本更像命中时，不该给出零漏拒阈值"


def test_样本文件坏形状会被拒(work_dir):
    """loader 要挡住缺字段、认不出的 expect 与重复 id，不要带病往下跑。"""
    bad_cases = {
        "缺 question": {"samples": [{"id": "a", "question": "", "expect": "hit", "source": "甲"}]},
        "认不出 expect": {"samples": [{"id": "a", "question": "问", "expect": "maybe"}]},
        "hit 缺 source": {"samples": [{"id": "a", "question": "问", "expect": "hit"}]},
        "id 重复": {
            "samples": [
                {"id": "a", "question": "问一", "expect": "refuse"},
                {"id": "a", "question": "问二", "expect": "refuse"},
            ]
        },
    }
    for name, payload in bad_cases.items():
        path = work_dir / "bad.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_samples(path)
