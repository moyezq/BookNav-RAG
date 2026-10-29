"""
看 json_object 打开时模型是一个对象还是多个。用来判断意图层要不要换成花括号配对、取第一段还是最后一段。不改解析代码。

原理（按系统流程）
  线上分类和抽标签都走 HttpClient.complete(json_object=True)。
  本脚本对分类提示、标签提示各打几次，数 content 里有几个配对的 JSON 对象。
  不跑召回、不改 classify.py。

  例子
    分类问「后视镜怎么折叠」→ 期望一个对象、贪婪正则与第一段相同。

怎么跑（本文件入口）
  职责：打几次对话模型，打印对象个数和贪婪正则能不能整段解析。
  前置：仓库根 .env 已有密钥、地址、模型名。
  产出：控制台。
  跑法：仓库根 uv run python scripts/probe/01_json_objects.py
"""

from __future__ import annotations

import faulthandler
import json
import re
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT；读 .env

from backend.conf import CLASSIFY_PROMPT, QUERY_TAG_PROMPT
from service.deepseek.http_client import HttpClient, ReplyKind

RUN_LIMIT_SEC: Final[int] = 300

CASES: Final[list[tuple[str, str, bool]]] = [
    ("分类", CLASSIFY_PROMPT.format(history_block="", question="后视镜怎么折叠"), True),
    ("分类", CLASSIFY_PROMPT.format(history_block="", question="你好"), True),
    (
        "标签",
        QUERY_TAG_PROMPT.format(
            catalog="来源: 演示手册\n标签: 主题=折叠",
            history_block="",
            question="后视镜怎么折叠",
        ),
        True,
    ),
    ("分类无json_object", CLASSIFY_PROMPT.format(history_block="", question="支架怎么折叠"), False),
]


def paired_objects(raw: str) -> list[str]:
    """花括号配对扫出的对象，字符串里的括号不计。"""
    candidates = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(raw or ""):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(raw[start : index + 1])
                start = -1
    return candidates


def judge(name: str, prompt: str, json_object: bool) -> None:
    """打一次，打印对象个数、贪婪正则、第一段和最后一段是否同一对象。"""
    reply = HttpClient().complete(prompt, json_object=json_object)
    raw = reply.content or ""
    objects = paired_objects(raw)
    greedy = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    greedy_text = greedy.group(0) if greedy else ""
    greedy_ok = False
    if greedy_text:
        try:
            json.loads(greedy_text)
            greedy_ok = True
        except json.JSONDecodeError:
            greedy_ok = False
    first = objects[0] if objects else ""
    last = objects[-1] if objects else ""
    print(f"问  {name} json_object={json_object}", flush=True)
    print(f"kind={reply.kind} 对象数={len(objects)} 贪婪能解析={greedy_ok}", flush=True)
    print(f"第一段==最后一段 {first == last}", flush=True)
    print(f"贪婪==第一段 {greedy_text == first} 贪婪==最后一段 {greedy_text == last}", flush=True)
    print(f"content {raw[:200]!r}", flush=True)
    if len(objects) <= 1 and greedy_ok:
        print("结论: 只有一个对象，取第一段、最后一段、贪婪正则结果相同", flush=True)
    elif len(objects) > 1 and not greedy_ok:
        print("结论: 多个对象，贪婪正则会跨段失败，要定取第一段还是最后一段", flush=True)
    elif len(objects) > 1:
        print("结论: 多个对象且贪婪也能解析，取第一段还是最后一段会不一样", flush=True)
    else:
        print("结论: 没有完整对象或 kind 不是 stop", flush=True)
    print("", flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")
    for name, prompt, json_object in CASES:
        judge(name, prompt, json_object)
