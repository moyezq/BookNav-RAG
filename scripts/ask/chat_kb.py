"""
多轮命令行：同一 session 循环提问，看跟进、筛选沿用、闲聊再回到检索。
不经问答页。空行或 q 结束。不评测 Hit@k。

原理（按系统流程）
  线上多轮：同一 session 的 turns 只给意图识别。检索问句可改写；筛选可沿用上一轮。
  本脚本开库一次，自造一份 session，每行输入问一句，invoke 同一张图。

  例子
    「后视镜怎么折叠」→ 检索，出答案和引用。
    「它怎么展开」→ 跟进，筛选沿用，仍答后视镜。
    「你好」→ 闲聊，无引用。
    「资料来源是哪份」→ 身份追问，走检索。

怎么跑（本文件入口；编号和上面例子不是同一套）
  职责：开库一次，固定 session，循环 input 直到空行或 q。
  入口步骤：
    1. open_knowledge_base — 读 backend/data/lancedb，只做一次。
    2. 循环：input → run_ask_graph（带 session）→ print_turn。
  输入：键盘；不覆盖意图标签。
  前置：已入库；python -m service.embed.server 已起；生成需 Chat API key。
  产出：控制台每一轮一块「结果」；召回调试仍写 backend/data/result/qa/step2_hits.json（只留最后一条）。
  对外：无。print_turn 在同目录 query_kb.py。
  跑法：仓库根 uv run python scripts/ask/chat_kb.py
        交互对话可能超过默认 5 分钟，RUN_LIMIT_SEC=3600。
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path
from typing import Final
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT 在 conf.base；读 yaml 与 .env

from backend.online.graph import run_ask_graph
from backend.online.recall.retrieve import open_knowledge_base
from scripts.ask.query_kb import print_turn

RUN_LIMIT_SEC: Final[int] = 3600
QUIT_WORDS: Final[frozenset[str]] = frozenset({"q", "quit", "exit"})


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    session = str(uuid4())
    print("多轮对话", flush=True)
    print("空行或 q 结束", flush=True)
    print(f"session  {session}", flush=True)

    # 1. LanceDB chunks 表 → 一份知识库
    knowledge_base = open_knowledge_base()
    print("[1] 已开库", flush=True)

    turn = 0
    while True:
        try:
            raw = input("\n问: ")
        except (EOFError, KeyboardInterrupt):
            print("\n结束", flush=True)
            break
        question = raw.strip()
        if not question or question.lower() in QUIT_WORDS:
            print("结束", flush=True)
            break
        turn += 1
        # 2. 同一 session 再跑意图 → 召回 → 生成
        print(f"[2] [{turn}] {question}", flush=True)
        result = run_ask_graph(knowledge_base, question, session=session)
        print_turn(question, result, turn=turn)
