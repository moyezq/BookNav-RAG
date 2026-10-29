"""
看换模型名时还要不要带 response_format json_object。不改 HttpClient。

原理（按系统流程）
  线上固定 CHAT_MODEL_NAME，分类和抽标签都带 json_object。
  本脚本对 .env 里的模型、以及接口报过错的另一个模型名，各打一次带 json_object 的请求。
  不改环境文件。

  例子
    deepseek-flash 带 json_object → 期望 stop 且 value 是对象。

怎么跑（本文件入口）
  职责：临时改进程里的 CHAT_MODEL_NAME，打完复原。
  前置：.env 已配密钥和地址。
  产出：控制台。
  跑法：仓库根 uv run python scripts/probe/04_json_mode_model.py
"""

from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT；读 .env

from service.deepseek.http_client import HttpClient, ReplyKind

RUN_LIMIT_SEC: Final[int] = 300
PROMPT: Final[str] = '用 json 回答。例如 {"sum": 2}。问题：1+1 等于几？'
MODELS: Final[list[str]] = [
    os.getenv("CHAT_MODEL_NAME", "").strip() or "deepseek-flash",
    "deepseek-v4-pro",
]


def probe(model: str) -> None:
    """指定模型名打一次 json_object。"""
    os.environ["CHAT_MODEL_NAME"] = model
    reply = HttpClient().complete(PROMPT, json_object=True)
    print(f"模型 {model}", flush=True)
    print(
        f"kind={reply.kind} status={reply.status_code} value={reply.value} err={reply.error_message[:160]!r}",
        flush=True,
    )
    if reply.kind == ReplyKind.STOP and isinstance(reply.value, dict):
        print("结论: 这个模型名带 json_object 能用，不必关", flush=True)
    elif reply.kind == ReplyKind.HTTP_ERROR:
        print("结论: 这个模型名带 json_object 被拒绝，换到它就要关或换提示", flush=True)
    else:
        print("结论: 不是 stop 也不是明确的 HTTP 拒绝", flush=True)
    print("", flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")
    original = os.getenv("CHAT_MODEL_NAME", "")
    seen = []
    try:
        for model in MODELS:
            if model in seen:
                continue
            seen.append(model)
            probe(model)
    finally:
        os.environ["CHAT_MODEL_NAME"] = original
