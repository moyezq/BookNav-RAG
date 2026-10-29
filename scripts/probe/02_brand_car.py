"""
看「哪个品牌」「什么车型」是不是身份追问，以及改写还会不会把它们原样留下。

原理（按系统流程）
  线上：分类不把品牌、车型当身份；有检索历史且命中「哪份 / 资料来源」才沿用上一轮检索问句。
  改写「原样输出」只覆盖来源、哪份文档。
  本脚本先直接调分类和改写，再对已启动的问答口走一轮检索再追问。

  例子
    先问折叠，再问「哪个品牌」→ 若途是检索且检索问句变成品牌，说明改写提示仍在起作用。

怎么跑（本文件入口）
  职责：分类 / 改写各打几句，再 POST 8031 两轮。
  前置：问答服务已起；.env 已配密钥。
  产出：控制台。
  跑法：仓库根 uv run python scripts/probe/02_brand_car.py
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path
from typing import Final

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT；读 .env

from backend.online.intent.classify import classify_route, format_turns, is_identity_question
from backend.online.intent.rewrite import rewrite_query

RUN_LIMIT_SEC: Final[int] = 300
ASK_URL: Final[str] = "http://127.0.0.1:8031/api/ask"
SESSION: Final[str] = "probebrand01"
HISTORY: Final[str] = format_turns(
    [
        {
            "question": "后视镜怎么折叠",
            "answer": "按下折叠开关。",
            "route": "retrieve",
        }
    ]
)


def ask(question: str) -> dict:
    """POST 已启动的问答口一轮。"""
    response = requests.post(
        ASK_URL,
        json={"text": question, "session": SESSION},
        timeout=120,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text}
    return {"status": response.status_code, "body": body}


def print_ask(question: str, payload: dict) -> None:
    """打问答口这一轮的答案和引用条数。"""
    body = payload["body"]
    if not isinstance(body, dict):
        print(f"问  {question}\nHTTP {payload['status']} {body!r}\n", flush=True)
        return
    answer = str(body.get("answer") or "")
    citations = body.get("citations") or []
    print(f"问  {question}", flush=True)
    print(f"HTTP {payload['status']} 引用 {len(citations)} 答 {answer[:180]!r}", flush=True)
    if citations:
        first = citations[0]
        print(f"首条引用 {first.get('source')} {first.get('snippet', '')[:80]!r}", flush=True)
    print("", flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. 身份标记：品牌、车型不应命中；哪份应命中
    for question in ("哪个品牌", "什么车型", "哪份文档", "资料来源是哪"):
        hit = is_identity_question(question)
        print(f"身份标记 {question!r} → {hit}", flush=True)
    print("", flush=True)

    # 2. 分类：无历史
    for question in ("哪个品牌", "什么车型", "哪份文档"):
        route = classify_route(question, "")
        print(f"分类无历史 {question!r} → {route}", flush=True)
    print("", flush=True)

    # 3. 改写：有检索历史。品牌、车型不应再被提示成身份原样输出
    for question in ("哪个品牌", "什么车型", "哪份文档"):
        rewritten = rewrite_query(question, HISTORY)
        print(f"改写有历史 {question!r} → {rewritten!r}", flush=True)
    print("", flush=True)

    # 4. 已启动的问答口：先检索，再追问品牌/车型/哪份
    alive = requests.get("http://127.0.0.1:8031/", timeout=5)
    print(f"[4] 问答口 {alive.status_code} {alive.text[:80]}", flush=True)
    print_ask("后视镜怎么折叠", ask("后视镜怎么折叠"))
    print_ask("哪个品牌", ask("哪个品牌"))
    print_ask("什么车型", ask("什么车型"))
    print_ask("哪份文档", ask("哪份文档"))
