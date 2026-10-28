"""
单轮快测：线上问答整条含生成，只问顶部 QUESTION 这一句，打出答案后退出。
不评测 Hit@k，不多轮。

原理（按系统流程）
  线上完整链路是：意图 → 召回 → 重排 → 同节邻块 → 拒答或生成。
  本脚本开库一次，把 QUESTION 整条跑完（含生成），打一块问/途/答/引后退出。

  例子（问句就是下面的 QUESTION；换库会变）
    「后视镜怎么折叠」→ 途=检索，侧栏应有引用来源与页。
    FILTER_SOURCE 非空 → 强制只搜该来源。

怎么跑（本文件入口；编号和上面「一条样本」不是同一套）
  职责：开库一次，invoke 图一次，打一块结果后退出。
  入口步骤：
    1. open_knowledge_base — 读 data/lancedb，只做一次。
    2. run_ask_graph — 意图 → 召回 → 重排 → 同节邻块 → 拒答或生成。
  输入：下面常量 QUESTION；FILTER_* 非空则覆盖意图标签。
  前置：已入库；python -m embed.server 已起；生成需 Chat API key。
  产出：控制台「结果」块；召回调试仍写 data/result/qa/step2_hits.json。
  对外：print_turn。
  跑法：仓库根 uv run python scripts/ask/query_kb.py
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import conf  # ROOT 在 conf.base；读 yaml 与 .env

from online.graph import run_ask_graph
from online.intent.classify import ROUTE_CHITCHAT
from online.recall.retrieve import open_knowledge_base

# ========== 按需修改 ==========
QUESTION: Final[str] = "后视镜怎么折叠"
FILTER_SOURCE: Final[str] = ""
FILTER_TAGS: Final[list[dict]] = []
RUN_LIMIT_SEC: Final[int] = 300
# ==============================


def default_filters() -> dict:
    """把顶部 FILTER_SOURCE / FILTER_TAGS 收成检索过滤；空 source 表示不限来源。"""
    return {"source": FILTER_SOURCE, "tags": FILTER_TAGS}


def print_turn(question: str, result: dict, *, turn: int | None = None) -> None:
    """
    把一轮生成结果打成可读的命令行块。

    功能：问、途、检索问句（有改写才打）、标、答、引。
    应用场景：query_kb 单轮；chat_kb 每一轮。
    输入：question — 用户原文；result — run_ask_graph 返回值；turn — 多轮序号，单轮不传。
    输出：无。打到控制台。
    """
    if turn is None:
        print("\n======== 结果 ========", flush=True)
    else:
        print(f"\n======== {turn} ========", flush=True)
    print(f"问  {question}", flush=True)
    way = "闲聊" if result["route"] == ROUTE_CHITCHAT else "检索"
    print(f"途  {way}", flush=True)
    text = result["text"]
    if text != question:
        print(f"检索问句  {text}", flush=True)
    tags = result["tags"]
    if tags:
        chips = "  ".join(f"{tag['field']}={tag['value']}" for tag in tags)
        print(f"标  {chips}", flush=True)
    else:
        print("标  无", flush=True)
    print("答", flush=True)
    answer_lines = result["answer"].splitlines() or [""]
    for line in answer_lines:
        print(f"    {line}", flush=True)
    citations = result["citations"]
    if not citations:
        print("引  无", flush=True)
        return
    print("引", flush=True)
    for index, citation in enumerate(citations, start=1):
        pages = citation["pages"]
        page_part = f" p.{','.join(str(page) for page in pages)}" if pages else ""
        snippet = " ".join(citation["snippet"].split())
        if len(snippet) > 80:
            snippet = snippet[:80] + "…"
        print(f"    {index}. {citation['source']}{page_part}", flush=True)
        print(f"       {snippet}", flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. LanceDB chunks 表 → 一份知识库
    knowledge_base = open_knowledge_base()
    print("[1] 已开库", flush=True)
    # 2. 图内意图 → 召回 → 生成或拒答
    result = run_ask_graph(knowledge_base, QUESTION, default_filters())
    print("[2] 已生成", flush=True)
    print_turn(QUESTION, result)
