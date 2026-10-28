"""
评测口：顺着线上检索路径走到同节邻块为止，判 Hit@k、拒答与闲聊路由。
精排之后的生成不跑。

原理（按系统流程）
  线上完整流程是：
    意图（分类 / 改写 / 标签）→ 召回 → 重排（截成 TOP_K）→ 同节邻块 → 拒答或生成。
  本脚本每条样本走到「同节邻块」结束，拿到一份和线上生成窗口同形状的命中列表，立刻判对错，然后下一条。
  不组提示、不调 Chat 写答案。所以测的是检索排位，不是模型会不会只看前几条、忽视后面。

  三种 expect 各自怎么判：
    hit    — 命中的 source 等于期望来源，且命中块的 pages 与期望 pages 有交集（块级页码）。
             正确答案在第 1 位 → Hit@1（同时也是 Hit@TOP_K）；在 2～TOP_K 位 → 只算 Hit@TOP_K。
             本脚本「通过」看 Hit@TOP_K，Hit@1 只打印。
    refuse — 不看来源和页，看 should_refuse 会不会拒答（列表空，或最好的向量余弦低于阈值）。
    chitchat — 路由必须是闲聊且不召回。这是意图层的判据，不涉及检索质量。

  页判据的判据：命中自带的 metadata.pages 是块级页码，所以「任一页相交即算中」不再必然成立。
  这个判据可不可信，先跑 scripts/ingest/verify_ingest.py 看页精度那一项。

  样本文件里跳过的两类（跳过数进汇总，不算未过）：
    - requires_sources 里有库内没有的来源；
    - hit 样本的期望来源不在库内。

  验收本脚本是否真在线上路径上：ROUTE_OVERRIDE="chitchat" 时正样本必须全部未命中。
  若仍报 Hit@1，说明本脚本没走线上路径。

怎么跑（本文件入口；编号和上面「一条样本」不是同一套）
  职责：加载与问答同一份库，按 SAMPLES_PATH 把每条样本的原理前几步跑完再判，然后退出。
  入口步骤：
    1. load_samples — 读样本文件，整个脚本只做一次。
    2. open_knowledge_base — 读 data/lancedb，只做一次。
    3. check_samples — 逐条：意图 → 召回 → 精排 → 同节邻块，再按 expect 判。
    4. CALIBRATE=True 时 calibrate_threshold — 扫拒答阈值。
  输入：下面常量 SAMPLES_PATH / ROUTE_OVERRIDE / SAMPLE_LIMIT / STRICT / CALIBRATE。
  前置：已入库；python -m embed.server 已起。分类与改写开时要 DEEPSEEK_API_KEY
        （ROUTE_OVERRIDE 注入时不需分类，改写仍要）。
  产出：控制台分组汇总与阈值扫点；退出码 0=按 STRICT 的判据通过，1=未过。不写进磁盘。
  对外：_hit_at、load_samples、check_samples、calibrate_threshold。
  跑法：仓库根 uv run python scripts/eval/eval_kb.py
        带改写的多样本可能超过 5 分钟，RUN_LIMIT_SEC=1800。
"""

from __future__ import annotations

import faulthandler
import json
import sys
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import conf  # ROOT 在 conf.base；读 yaml 与 .env

from online.generate.answer import best_vector_score, should_refuse
from online.graph import ask_once_core
from online.recall.retrieve import KnowledgeBase, open_knowledge_base
from online.intent.classify import ROUTE_CHITCHAT
from conf import REFUSE_MIN_SCORE, TOP_K, USE_REWRITE

# ========== 按需修改 ==========
SAMPLES_PATH: Final[Path] = Path(__file__).resolve().parent / "eval_samples.json"
# 本机大集：改成同目录 eval_samples.private.json
ROUTE_OVERRIDE: Final[Literal["", "retrieve", "chitchat"]] = ""
SAMPLE_LIMIT: Final[int] = 0  # 只跑前几条，0 为全部
STRICT: Final[bool] = False
CALIBRATE: Final[bool] = False
RUN_LIMIT_SEC: Final[int] = 1800
# ==============================

EMPTY_FILTERS: Final[dict] = {"source": "", "tags": []}
CALIBRATE_LOW: Final[float] = 0.10
CALIBRATE_HIGH: Final[float] = 0.90
CALIBRATE_STEP: Final[float] = 0.01


def load_samples(path: Path) -> dict:
    """
    读样本文件。

    功能：认 corpus_sources / notes / samples；每条样本最少要有 id、question、expect。
    应用场景：脚本入口第 1 步。
    输入：path — eval_samples.json 路径。
    输出：整个 JSON dict；samples 为 list。
    异常：文件不存在 → FileNotFoundError；缺字段或 expect 不认识 → ValueError。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} 里没有 samples 列表")
    known = {"hit", "refuse", "chitchat"}
    seen = set()
    for index, sample in enumerate(samples, start=1):
        for key in ("id", "question", "expect"):
            if not str(sample.get(key, "")).strip():
                raise ValueError(f"{path} 第 {index} 条缺 {key}")
        if sample["expect"] not in known:
            raise ValueError(f"{path} 第 {index} 条 expect={sample['expect']!r} 不认识，只认 {sorted(known)}")
        if sample["id"] in seen:
            raise ValueError(f"{path} 里 id 重复：{sample['id']}")
        seen.add(sample["id"])
        if sample["expect"] == "hit" and not str(sample.get("source", "")).strip():
            raise ValueError(f"{path} 第 {index} 条 hit 样本缺 source")
    return payload


def _pages_overlap(hit_pages: Sequence[int], expected_pages: Sequence[int]) -> bool:
    """命中页与期望页是否有交集。"""
    return bool(set(hit_pages) & set(expected_pages))


def _hit_at(hits: list, top_k: int, source: str, pages: list) -> bool:
    """
    前 top_k 条是否含期望来源，可选再对页。

    功能：Hit@k。有 pages 则来源命中里还要页码相交。
    应用场景：正样本判定。
    输入：hits — 检索命中；top_k — 窗口；source — 期望来源；pages — 可空。
    输出：是否命中。
    """
    window = hits[:top_k]
    source_hits = [hit for hit in window if hit["source"] == source]
    if not source_hits:
        return False
    if pages:
        return any(_pages_overlap(hit["pages"], pages) for hit in source_hits)
    return True


def _skip_reason(sample: dict, kb_sources: set) -> str:
    """库内缺来源就跳过；返回空串表示要跑这条。"""
    missing = [name for name in sample.get("requires_sources", []) if name not in kb_sources]
    if missing:
        return "库内缺来源 " + "、".join(missing)
    if sample["expect"] == "hit" and sample["source"] not in kb_sources:
        return f"库内缺期望来源 {sample['source']}"
    return ""


def check_samples(
    knowledge_base: KnowledgeBase,
    samples: list[dict],
    *,
    route_override: str = "",
    limit: int = 0,
) -> list[dict]:
    """
    对每条样本跑线上检索核心，对照 Hit@k、拒答或闲聊路由。

    功能：不过滤成期望来源；按 expect 分别判；库内缺来源的样本跳过并记原因。
    应用场景：入口第 3 步；CALIBRATE 复用同一批结果。
    输入：knowledge_base — 已 load；samples — 样本列表；route_override — 注入路由（空则走真实分类）；
          limit — 只跑前几条，0 为全部。
    输出：[{id, group, question, expect, ok, skipped, reason, hits, rewritten, route, vector_score}, ...]。
    """
    kb_sources = set(knowledge_base.tag_catalog()["sources"])
    picked = samples[:limit] if limit else samples
    print(
        f"[3] USE_REWRITE={USE_REWRITE} TOP_K={TOP_K} REFUSE_MIN_SCORE={REFUSE_MIN_SCORE} "
        f"samples={len(picked)} 库内来源={len(kb_sources)} route_override={route_override or '无'}",
        flush=True,
    )
    rows = []
    for index, sample in enumerate(picked, start=1):
        question = sample["question"]
        expect = sample["expect"]
        group = sample.get("group", "未分组")
        label = f"[{index}/{len(picked)}] {sample['id']} {question!r} expect={expect} group={group}"
        skip = _skip_reason(sample, kb_sources)
        if skip:
            print(f"[3] {label}\n  跳过：{skip}", flush=True)
            rows.append(
                {
                    "id": sample["id"],
                    "group": group,
                    "question": question,
                    "expect": expect,
                    "ok": None,
                    "skipped": True,
                    "reason": skip,
                    "hits": [],
                    "rewritten": question,
                    "route": "",
                    "vector_score": None,
                }
            )
            continue

        print(f"[3] {label}", flush=True)
        core = ask_once_core(
            knowledge_base,
            question,
            turns=sample.get("turns"),
            filters=sample.get("filters", EMPTY_FILTERS),
            route_override=route_override,
            write_debug=False,
        )
        hits = core["hits"]
        rewritten = core["text"]
        if rewritten != question:
            print(f"  改写 → {rewritten!r}", flush=True)
        cosine = best_vector_score(hits)
        cosine_text = "无" if cosine is None else f"{cosine:.4f}"
        print(
            f"  路由={core['route']} 筛选={core['filters']} vector_score={cosine_text} hits={len(hits)}",
            flush=True,
        )

        if expect == "refuse":
            ok = should_refuse(hits, rewritten)
            reason = "应拒答且已拒" if ok else "应拒答但仍会生成"
        elif expect == "chitchat":
            ok = core["route"] == ROUTE_CHITCHAT and not hits
            reason = "判成闲聊且未召回" if ok else f"路由={core['route']} hits={len(hits)}，不是闲聊路径"
        else:
            at1 = _hit_at(hits, 1, sample["source"], sample.get("pages", []))
            atk = _hit_at(hits, TOP_K, sample["source"], sample.get("pages", []))
            ok = atk
            if at1:
                reason = "Hit@1"
            elif atk:
                reason = f"Hit@{TOP_K} 未进@1"
            else:
                reason = "未命中期望来源或页"

        print(f"  结论={'通过' if ok else '未过'} {reason}", flush=True)
        rows.append(
            {
                "id": sample["id"],
                "group": group,
                "question": question,
                "expect": expect,
                "ok": ok,
                "skipped": False,
                "reason": reason,
                "hits": hits,
                "rewritten": rewritten,
                "route": core["route"],
                "vector_score": cosine,
            }
        )
    return rows


def _print_group_summary(rows: list[dict]) -> None:
    """按 group 打一张汇总表：跑了几条、过几条、正样本的 Hit@1/@k 各几条。"""
    groups: OrderedDict[str, dict] = OrderedDict()
    for row in rows:
        stat = groups.setdefault(
            row["group"],
            {"total": 0, "skipped": 0, "passed": 0, "hit_n": 0, "hit1_n": 0, "hitk_n": 0},
        )
        stat["total"] += 1
        if row["skipped"]:
            stat["skipped"] += 1
            continue
        if row["expect"] == "hit":
            stat["hit_n"] += 1
            if row["ok"]:
                stat["hitk_n"] += 1
            if row["reason"] == "Hit@1":
                stat["hit1_n"] += 1
        if row["ok"]:
            stat["passed"] += 1
    print("[3] 分组汇总", flush=True)
    print(f"  {'group':<22} {'跑':>3} {'过':>3} {'跳过':>4} {'Hit@1':>7} {'Hit@k':>7}", flush=True)
    for name, stat in groups.items():
        print(
            f"  {name:<22} {stat['total']:>3} {stat['passed']:>3} {stat['skipped']:>4} "
            f"{stat['hit1_n']:>3}/{stat['hit_n']:<3} {stat['hitk_n']:>3}/{stat['hit_n']:<3}",
            flush=True,
        )


def calibrate_threshold(rows: list[dict]) -> dict:
    """
    扫拒答阈值，给拒答阈值定依据。

    功能：把「该拒答的样本」与「该回答的正样本」的最好多路命中拿去扫阈值，判据用线上同一个 should_refuse。
          除余弦阈值外，空命中与词面兜底判据照旧生效，所以扫出来的是「真实线上行为下的」工作点。
    应用场景：定阈值；README 填数字。
    输入：rows — check_samples 的返回。
    输出：{negatives, positives, rows: [...], current, best_f1, best_recall}；没有负样本或没有正样本时 rows 为空。
          best_f1 — F1 最高的点；best_recall — 零漏拒的点里精确率最高的那个（并列取更高阈值）。
          best_recall 为 None 说明负样本里有比正样本余弦更高的，任何阈值都漏拒，先查样本而不是调阈值。
    """
    judged = [
        row
        for row in rows
        if not row["skipped"] and row["expect"] in {"hit", "refuse"}
    ]
    negatives = [row for row in judged if row["expect"] == "refuse"]
    positives = [row for row in judged if row["expect"] == "hit"]
    if not negatives or not positives:
        print(
            f"[4] 阈值校准需要正负样本都有：负 {len(negatives)} 条 / 正 {len(positives)} 条，跳过校准",
            flush=True,
        )
        return {"negatives": len(negatives), "positives": len(positives), "rows": []}

    neg_cosines = sorted(row["vector_score"] for row in negatives if row["vector_score"] is not None)
    pos_cosines = sorted(row["vector_score"] for row in positives if row["vector_score"] is not None)
    if neg_cosines:
        print(
            f"[4] 负样本最高余弦 {neg_cosines[-1]:.4f} / 最低 {neg_cosines[0]:.4f}（{len(neg_cosines)} 条带向量分）",
            flush=True,
        )
    if pos_cosines:
        print(
            f"[4] 正样本最高余弦 {pos_cosines[-1]:.4f} / 最低 {pos_cosines[0]:.4f}（{len(pos_cosines)} 条带向量分）",
            flush=True,
        )

    sweep = []
    step = CALIBRATE_STEP
    point = CALIBRATE_LOW
    while point <= CALIBRATE_HIGH + 1e-9:
        threshold = round(point, 2)
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for row in judged:
            predicted = should_refuse(row["hits"], row["rewritten"], min_score=threshold)
            if row["expect"] == "refuse":
                if predicted:
                    true_positive += 1
                else:
                    false_negative += 1
            elif predicted:
                false_positive += 1
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
        point += step

    by_f1 = sorted(sweep, key=lambda item: (item["f1"], item["precision"]), reverse=True)
    no_miss = [item for item in sweep if item["false_negative"] == 0]
    best_recall = max(no_miss, key=lambda item: (item["precision"], item["threshold"])) if no_miss else None
    current = min(sweep, key=lambda item: abs(item["threshold"] - REFUSE_MIN_SCORE))
    print(f"[4] 阈值扫点 {CALIBRATE_LOW:.2f}～{CALIBRATE_HIGH:.2f}（负 {len(negatives)} / 正 {len(positives)}）", flush=True)
    print(f"  {'阈值':>5} {'拒答精确率':>10} {'拒答召回率':>10} {'F1':>6} {'误拒正样本':>10} {'漏拒负样本':>10}", flush=True)
    for item in by_f1[:3]:
        print(
            f"  {item['threshold']:>5.2f} {item['precision']:>10.3f} {item['recall']:>10.3f} "
            f"{item['f1']:>6.3f} {item['false_positive']:>10} {item['false_negative']:>10}",
            flush=True,
        )
    print(
        f"  当前 conf={REFUSE_MIN_SCORE:.2f}：精确率 {current['precision']:.3f} "
        f"召回率 {current['recall']:.3f} 误拒 {current['false_positive']} 漏拒 {current['false_negative']}",
        flush=True,
    )
    if best_recall is not None:
        print(
            f"  零漏拒里精确率最高的阈值 {best_recall['threshold']:.2f}：精确率 {best_recall['precision']:.3f} "
            f"召回率 {best_recall['recall']:.3f} 误拒 {best_recall['false_positive']}",
            flush=True,
        )
    else:
        print("  任何阈值都有漏拒：负样本里有比正样本余弦更高的，先看这些样本对不对", flush=True)
    return {
        "negatives": len(negatives),
        "positives": len(positives),
        "rows": sweep,
        "current": current,
        "best_f1": by_f1[0],
        "best_recall": best_recall,
    }


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. 读样本文件
    payload = load_samples(SAMPLES_PATH)
    print(f"[1] 已读 {SAMPLES_PATH} samples={len(payload['samples'])}", flush=True)
    # 2. 加载与问答同一份库
    knowledge_base = open_knowledge_base()
    print("[2] 已开库", flush=True)
    # 3. 逐条跑线上检索核心，再判 Hit@k / 拒答 / 闲聊
    rows = check_samples(
        knowledge_base,
        payload["samples"],
        route_override=ROUTE_OVERRIDE,
        limit=SAMPLE_LIMIT,
    )
    _print_group_summary(rows)
    ran = [row for row in rows if not row["skipped"]]
    failed = [row for row in ran if not row["ok"]]
    print(f"[3] 通过 {len(ran) - len(failed)} / {len(ran)}（跳过 {len(rows) - len(ran)}）", flush=True)
    if CALIBRATE:
        # 4. 扫拒答阈值
        calibrate_threshold(rows)
    if STRICT and failed:
        print(f"[3] STRICT：{len(failed)} 条未过", flush=True)
        raise SystemExit(1)
    raise SystemExit(0)
