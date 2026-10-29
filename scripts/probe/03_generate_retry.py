"""
看生成失败后再打一次同一请求能不能变成 stop。不往产品代码里加重试。

原理（按系统流程）
  线上生成失败（length / http_error / 未配置）直接给固定句，不重试。
  待办是「失败后再请求一次」。
  本脚本对同一提示连打两次：一次正常 max_tokens，一次故意截断后再用同样截断参数重打。
  不改 answer.py。

  例子
    max_tokens=8 的 JSON 两次都是 length → 同样参数重试救不了截断。

怎么跑（本文件入口）
  职责：HttpClient 连打，打印两次 kind。
  前置：.env 已配密钥。
  产出：控制台。
  跑法：仓库根 uv run python scripts/probe/03_generate_retry.py
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT；读 .env

from service.deepseek.http_client import HttpClient, ReplyKind

RUN_LIMIT_SEC: Final[int] = 300
TEXT_PROMPT: Final[str] = "用一句话回答：1+1 等于几？"
JSON_PROMPT: Final[str] = '用 json 回答。例如 {"sum": 2, "note": "说明"}。问题：1+1 等于几？note 写满三十个字。'


def twice(prompt: str, *, json_object: bool, max_tokens: int) -> tuple:
    """同一提示、同一 max_tokens 连打两次。"""
    saved = HttpClient.MAX_TOKENS
    HttpClient.MAX_TOKENS = max_tokens
    try:
        first = HttpClient().complete(prompt, json_object=json_object)
        second = HttpClient().complete(prompt, json_object=json_object)
    finally:
        HttpClient.MAX_TOKENS = saved
    return first, second


def report(title: str, first, second) -> None:
    """打印两次 kind，并说重试有没有把非 stop 救回来。"""
    print(f"== {title} ==", flush=True)
    print(f"第1次 kind={first.kind} content={first.content[:80]!r} err={first.error_message[:80]!r}", flush=True)
    print(f"第2次 kind={second.kind} content={second.content[:80]!r} err={second.error_message[:80]!r}", flush=True)
    if first.kind != ReplyKind.STOP and second.kind == ReplyKind.STOP:
        print("结论: 同样参数再打一次，第二次变成 stop，重试有用", flush=True)
    elif first.kind != ReplyKind.STOP and second.kind != ReplyKind.STOP:
        print("结论: 两次都不是 stop，同样参数重试救不了这次失败", flush=True)
    else:
        print("结论: 第一次已经是 stop，看不出重试是否必要", flush=True)
    print("", flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. 正常生成：两次都 round-trip
    report("正常文本 max_tokens=2048", *twice(TEXT_PROMPT, json_object=False, max_tokens=2048))
    # 2. 故意截断：两次同一上限
    report("JSON 截断 max_tokens=8", *twice(JSON_PROMPT, json_object=True, max_tokens=8))
    # 3. 截断之后用正常上限再打一次（不是产品里的「同样请求重试」）
    saved = HttpClient.MAX_TOKENS
    HttpClient.MAX_TOKENS = 8
    cut = HttpClient().complete(JSON_PROMPT, json_object=True)
    HttpClient.MAX_TOKENS = saved
    recovered = HttpClient().complete(JSON_PROMPT, json_object=True)
    print("== 先截断再换回 2048 ==", flush=True)
    print(f"截断 kind={cut.kind} 之后 kind={recovered.kind}", flush=True)
    if cut.kind == ReplyKind.LENGTH and recovered.kind == ReplyKind.STOP:
        print("结论: 加大 max_tokens 能从 length 恢复；产品若按原参数重试则不是这条路", flush=True)
    print("", flush=True)
