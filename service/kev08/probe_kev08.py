# -*- coding: utf-8 -*-
"""
看清 Kev 到底吐什么：请求原文、响应原文、以及我们这一侧做了哪些处理。

职责：两个中文用例，一问「证据够不够」，只差中间那段原文；每条按顺序打印四块：
      1. 【输入】我们 POST 出去的完整 JSON
      2. 【模型打分的东西】按 kev/api.py 的渲染规则复刻的 state 与选项串（接口不回传，所以标为复刻）
      3. 【响应】HTTP 状态、request id、耗时、原样 body（不裁剪、不改字段）
      4. 【解析】怎么从 body 得到 noul 与 confidence，算式写出来
      最后给两条判定与两者的概率差（对比度）。
前置：服务已起（serve_kev08.py）。
输入：KEV_BASE_URL / KEV_API_KEY / KEV_MODEL 环境变量（默认 http://127.0.0.1:8034）。
输出：上述四块 + 汇总；退出码 0=两条判定都过，1=有失败或端点不可用。
怎么跑：uv run python probe_kev08.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8034"
TIMEOUT_SEC = 120
CONTRAST_MIN = 0.3  # 有证据与无证据的 noul 概率差；低于这个算「没拉开」
NOUL_MIN = 0.5  # noul 判「够」的阈值

EVIDENCE_QUESTION = "刹车油多久更换一次？"
EVIDENCE_YES = "4.2 制动系统。衬片厚度小于 3 毫米时更换前刹车片；剩余 2 毫米时磨损报警响起。无论里程多少，每 24 个月更换一次刹车油。"
EVIDENCE_NO = "7.1 车轮定位。在整备质量下把前束设为 0.1 度。调整前检查横拉杆端部间隙。"
ENOUGH_INSTRUCTIONS = "原文里有没有回答这个问题所需的事实？"


def evidence_state(passage: str) -> str:
    """把「问题 + 候选原文」拼成一段 state，供 noul 判断用。"""
    return f"Question: {EVIDENCE_QUESTION}\n\nPassage: {passage}"


def enough_question() -> dict:
    """noul：只看给的原文，够不够回答那个问题。不给 criteria，选项就是 no / yes。"""
    return {"enough": {"type": "noul", "instructions": ENOUGH_INSTRUCTIONS}}


# name, state, questions, 期望
CASES = (
    {"name": "中文·证据充分", "state": evidence_state(EVIDENCE_YES), "questions": enough_question(), "expect": ("noul_high", "enough")},
    {"name": "中文·证据缺失", "state": evidence_state(EVIDENCE_NO), "questions": enough_question(), "expect": ("noul_low", "enough")},
)


def render(value) -> str:
    """复刻 kev/api.py 的 render()：str 原样，dict 变 `key: value`，list 变 `- item`。"""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"- {render(item).lstrip()}" for item in value)
    return "\n".join(f"{k}: {render(v)}" if not isinstance(v, (dict, list)) else f"{k}:\n{render(v)}" for k, v in value.items())


def option_text(name: str, desc) -> str:
    """复刻 kev/api.py 的 option_text()：选项串是 `名字: 描述`，没描述就只有名字。"""
    return name if desc in (None, "") else f"{name}: {render(desc)}"


def model_view(state, questions: dict) -> str:
    """
    按 kev/api.py 的规则复刻「模型实际打分的东西」，用来对照我们以为的输入。

    功能：渲染 state；每个问题一行指令 + 一行选项串。
    应用场景：接口只回概率、不回选项文本，所以这份视图是推导出来的，不是服务端给的。
    输入：state — 请求里的 state；questions — 请求里的 questions。
    输出：多行字符串。
    """
    lines = [f"state（模型看到的第一段）:\n{render(state)}"]
    for qid, question in questions.items():
        if question["type"] == "noul":
            criteria = question.get("criteria") or {}
            options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
        elif question["type"] == "choice":
            options = [option_text(name, desc) for name, desc in question["criteria"].items()]
        else:
            options = [render(level) for level in question["criteria"]]
        lines.append(f"[{qid}] type={question['type']}  指令={render(question.get('instructions'))!r}")
        lines.append(f"    选项（{len(options)} 个，模型只在这些里打分）: {options}")
    return "\n".join(lines)


def call(base_url: str, headers: dict, payload: dict) -> tuple[requests.Response, float]:
    """发一条 systemone 请求，返回 (响应对象, 秒)。状态码不在这里判，交给调用方打印。"""
    started = time.perf_counter()
    response = requests.post(base_url + "/v1/systemone", json=payload, headers=headers, timeout=TIMEOUT_SEC)
    return response, time.perf_counter() - started


def explain(answer: dict) -> str:
    """把一条答案的来历写清楚：noul 是哪个概率，choice 的 argmax 与 confidence 算式。"""
    if answer.get("type") == "noul":
        return f"  noul 就是 p(true)：内部两个选项是 [no, yes]，接口把 yes 的概率单独回给你 → {answer.get('noul')}"
    if answer.get("type") == "choice":
        probabilities = answer.get("probabilities", {})
        top = max(probabilities, key=probabilities.get)
        k = len(probabilities)
        confidence = 1.0 if k == 1 else (probabilities[top] - 1 / k) / (1 - 1 / k)
        ranks = " > ".join(f"{name} {value:.4f}" for name, value in sorted(probabilities.items(), key=lambda kv: -kv[1]))
        return (
            f"  probabilities 排序: {ranks}\n"
            f"  choice = argmax = {top}（模型不会说「都不是」，它一定在这 {k} 个里挑）\n"
            f"  confidence = (p_max − 1/K) / (1 − 1/K) = ({probabilities[top]:.4f} − 1/{k}) / (1 − 1/{k}) = {confidence:.4f}"
        )
    return f"  score 是期望档位：E[level] = {answer.get('score')}"


def judge(answer: dict, expect: tuple) -> bool:
    """按期望判一条：choice 比选中的名字，noul_high/low 比 0.5。expect 允许 (kind, key) 或 (kind, key, 期望值)。"""
    kind = expect[0]
    wanted = expect[2] if len(expect) > 2 else None
    if kind == "choice":
        return answer.get("choice") == wanted
    if kind == "noul_high":
        return answer.get("noul", 0.0) >= NOUL_MIN
    return answer.get("noul", 1.0) < NOUL_MIN


def main() -> None:
    """打印端点信息，然后逐条走「输入 → 模型视图 → 原始响应 → 解析」，最后汇总。"""
    base_url = os.getenv("KEV_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("KEV_MODEL", "kev-latest")
    headers = {"Content-Type": "application/json"}
    if os.getenv("KEV_API_KEY", ""):
        headers["Authorization"] = "Bearer " + os.getenv("KEV_API_KEY", "")
    print(f"[端点] {base_url}  model={model}")

    card = requests.get(base_url + "/v1/models", headers=headers, timeout=TIMEOUT_SEC)
    print("[GET /v1/models] 服务自报的运行时（设备、精度、温度、上下文上限、缓存）:")
    print(card.text if card.status_code == 200 else f"HTTP {card.status_code}: {card.text[:300]}")

    passed = failed = 0
    noul_by_case: dict[str, float] = {}
    for index, case in enumerate(CASES, start=1):
        payload = {"model": model, "state": case["state"], "questions": case["questions"]}
        print(f"\n{'=' * 20} 用例 {index}/{len(CASES)}: {case['name']} {'=' * 20}")
        print("[输入] 我们 POST 的 body：")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("[模型打分的东西] 按 kev/api.py 规则复刻（接口不回传，供对照）:")
        print(model_view(case["state"], case["questions"]))

        response, elapsed = call(base_url, headers, payload)
        print(f"[响应] HTTP {response.status_code} · {elapsed * 1000:.0f} ms · request-id={response.headers.get('x-typesafe-request-id')}")
        print("[响应] 原始 body（原样，无加工）：")
        print(response.text)
        if response.status_code != 200:
            print("[失败] 非 200，停止。")
            sys.exit(1)

        answers = response.json().get("answers", {})
        print("[解析] 我们这一侧只做 JSON 解析，不改字段名、不重算概率：")
        for key, answer in answers.items():
            if answer.get("type") == "noul":
                noul_by_case[case["name"]] = answer.get("noul", float("nan"))
            print(f"  ── {key}")
            print(explain(answer))
            ok = judge(answer, case["expect"])
            passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
            print(f"  [判定] {'PASS' if ok else 'FAIL'}（期望：{case['expect']}；阈值 {NOUL_MIN}）")

    first = noul_by_case.get("中文·证据充分", float("nan"))
    second = noul_by_case.get("中文·证据缺失", float("nan"))
    contrast = first - second
    print(f"\n[汇总] 判定用例 {passed + failed} 条：通过 {passed}，失败 {failed}")
    print(f"[汇总] 同一问题、只换原文：有证据 {first:.4f} − 无证据 {second:.4f} = {contrast:+.4f}（门槛 {CONTRAST_MIN}）")
    print("[汇总] 两个数值都贴着 0.5 时，差值必然接近 0：那就是没有判别力，而不是「模型选了 no」。")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
