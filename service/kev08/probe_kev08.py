"""
读 probe_intent.json，打 POST /v1/systemone，看 kev08 对闲聊 / 结束 / 查库三分类准不准。不接提问口、不送召回原文。

原理（按系统流程）
  现网意图层用 LLM 把问句分成检索或闲聊；拒答发生在召回之后，本脚本不测拒答。
  本脚本只测决策接口能不能在意图层一次分成三档：chitchat / end / retrieve。
  end 是用户在收尾对话，不是域外拒答。每条样本：问句 → 一次 choice → 对照文件里的 expect。

  一条样本逐步发生什么：
    1. state = 文件里的 state_prefix + 问句；三分类题原样放进 questions。
    2. 看 answers.route.choice 是否等于 expect。
    3. 到此停止。汇总准确率与每类对错。

  例子（问句就是 probe_intent.json；一次实测以终端为准）
    「你好」→ chitchat。
    「再见」→ end。
    「支架怎么折叠」→ retrieve。

怎么跑（本文件入口；编号和上面「一条样本」不是同一套）
  职责：读样本文件，逐条 POST 再判，打印准确率后退出。
  入口步骤：
    1. load_dataset — 读并校验 probe_intent.json。
    2. main 其余 — 探活后逐条请求、汇总。
  输入：同目录 probe_intent.json；端点可用 KEV_BASE_URL / KEV_API_KEY / KEV_MODEL 覆盖。
  前置：本目录 serve_kev08.py 已起。
  产出：控制台；退出码 0=准确率达标且每类至少对一条，1=文件坏、端点不可用或准确率不够。
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8034"
TIMEOUT_SEC = 120
RUN_LIMIT_SEC = 300
MIN_ACCURACY = 0.7
SAMPLES_PATH = Path(__file__).resolve().parent / "probe_intent.json"
LABEL_TEXT = {"chitchat": "闲聊", "end": "结束", "retrieve": "查库"}


def load_dataset(path: Path) -> dict[str, Any]:
    """
    读意图样本文件并校验形状。

    功能：labels、三分类题、samples 的 id / question / expect 必须自洽。
    应用场景：探针启动一次。
    输入：path — 一般是同目录 probe_intent.json。
    输出：文件根对象。
    异常：缺文件、JSON 坏、缺字段或 expect 不在 labels → ValueError。
    """
    if not path.is_file():
        raise ValueError(f"找不到样本文件：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    question = payload.get("question") or {}
    samples = payload.get("samples")
    if not isinstance(labels, list) or set(labels) != set(LABEL_TEXT):
        raise ValueError(f"labels 必须是 {sorted(LABEL_TEXT)}，现在是 {labels!r}")
    prefix = payload.get("state_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("state_prefix 必须是非空字符串")
    for key in ("name", "type", "instructions", "criteria"):
        if key not in question:
            raise ValueError(f"question 缺少 {key}")
    if question["type"] != "choice":
        raise ValueError("question.type 必须是 choice")
    if set(question["criteria"]) != set(labels):
        raise ValueError("question.criteria 的键必须与 labels 一致")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples 不能空")
    ids: list[str] = []
    questions: list[str] = []
    expects: list[str] = []
    for sample in samples:
        for key in ("id", "group", "question", "expect"):
            if not str(sample.get(key, "")).strip():
                raise ValueError(f"样本缺 {key}：{sample!r}")
        if sample["expect"] not in labels:
            raise ValueError(f"{sample['id']} 的 expect={sample['expect']!r} 不在 labels 里")
        ids.append(sample["id"])
        questions.append(sample["question"])
        expects.append(sample["expect"])
    if len(ids) != len(set(ids)):
        raise ValueError("样本 id 有重复")
    if len(questions) != len(set(questions)):
        raise ValueError("样本问句有重复")
    missing = set(labels) - set(expects)
    if missing:
        raise ValueError(f"每一类至少要有一条样本，缺 {sorted(missing)}")
    return payload


def call_route(
    base_url: str,
    headers: dict[str, str],
    model: str,
    state: str,
    route_question: dict[str, Any],
    question_name: str,
) -> tuple[int, dict[str, Any], float]:
    """POST 一条意图三分类，返回 (HTTP 状态, 解析后的 body 或原文包装, 秒)。"""
    payload = {
        "model": model,
        "state": state,
        "questions": {question_name: route_question},
    }
    started = time.perf_counter()
    response = requests.post(
        base_url + "/v1/systemone",
        json=payload,
        headers=headers,
        timeout=TIMEOUT_SEC,
    )
    elapsed = time.perf_counter() - started
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        body = {"raw": response.text}
    return response.status_code, body, elapsed


def main() -> None:
    """读样本 → 探活 → 逐条判定 → 汇总准确率。"""
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        dataset = load_dataset(SAMPLES_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[失败] 样本文件：{error}", flush=True)
        sys.exit(1)

    samples: list[dict[str, Any]] = dataset["samples"]
    state_prefix = str(dataset["state_prefix"])
    route_spec: dict[str, Any] = dataset["question"]
    question_name = str(route_spec["name"])
    route_question = {
        "type": route_spec["type"],
        "instructions": route_spec["instructions"],
        "criteria": route_spec["criteria"],
    }

    base_url = os.getenv("KEV_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("KEV_MODEL", "kev-latest")
    headers = {"Content-Type": "application/json"}
    if os.getenv("KEV_API_KEY", ""):
        headers["Authorization"] = "Bearer " + os.getenv("KEV_API_KEY", "")
    print(f"[端点] {base_url}", flush=True)
    print(f"[样本] {SAMPLES_PATH.name} · {len(samples)} 条", flush=True)

    try:
        ping = requests.get(base_url + "/v1/models", headers=headers, timeout=TIMEOUT_SEC)
    except requests.RequestException as error:
        print(f"[失败] 连不上：{error}", flush=True)
        print("先在本目录起服务：uv run python serve_kev08.py", flush=True)
        sys.exit(1)
    if ping.status_code != 200:
        print(f"[失败] GET /v1/models HTTP {ping.status_code}: {ping.text[:300]}", flush=True)
        sys.exit(1)
    print("[探活] 服务在听", flush=True)
    print(f"[身份] {state_prefix!r}", flush=True)
    print(
        f"[题面] {question_name}：{' / '.join(LABEL_TEXT[name] for name in dataset['labels'])}",
        flush=True,
    )

    gold: list[str] = []
    pred: list[str] = []
    failures: list[str] = []
    for index, sample in enumerate(samples, start=1):
        expect = sample["expect"]
        status, body, elapsed = call_route(
            base_url,
            headers,
            model,
            state_prefix + sample["question"],
            route_question,
            question_name,
        )
        answer = (body.get("answers") or {}).get(question_name) or {}
        got = str(answer.get("choice") or "")
        probabilities = answer.get("probabilities") or {}
        if status != 200:
            got = f"HTTP {status}"
        gold.append(expect)
        pred.append(got)
        hit = status == 200 and got == expect
        mark = "PASS" if hit else "FAIL"
        expect_text = LABEL_TEXT.get(expect, expect)
        got_text = LABEL_TEXT.get(got, got)
        print(
            f"[{index}/{len(samples)}] {sample['id']}  {sample['question']}  "
            f"期望 {expect_text}  得到 {got_text}  {elapsed * 1000:.0f} ms  {mark}",
            flush=True,
        )
        if not hit:
            failures.append(
                f"{sample['id']} 问「{sample['question']}」期望 {expect_text} 得到 {got_text}  "
                f"p={probabilities}"
            )

    total = len(samples)
    correct = sum(a == b for a, b in zip(gold, pred))
    accuracy = correct / total if total else 0.0
    print(f"\n[汇总] 准确率 {correct}/{total} = {accuracy:.1%}（门槛 {MIN_ACCURACY:.0%}）", flush=True)

    gold_count = Counter(gold)
    hit_count: Counter[str] = Counter()
    for expect, got in zip(gold, pred):
        if expect == got:
            hit_count[expect] += 1
    collapsed = False
    for label in dataset["labels"]:
        n = gold_count[label]
        ok = hit_count[label]
        print(f"[汇总] {LABEL_TEXT[label]}  {ok}/{n}", flush=True)
        if n > 0 and ok == 0:
            collapsed = True

    if failures:
        print("[失败]", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
    if collapsed:
        print("[汇总] 有一类全错：模型塌成单一标签，三分类不可用。", flush=True)
    if accuracy < MIN_ACCURACY:
        print(f"[汇总] 准确率低于 {MIN_ACCURACY:.0%}，这批样本上还不能替 LLM 做意图。", flush=True)
    sys.exit(0 if accuracy >= MIN_ACCURACY and not collapsed else 1)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    main()
