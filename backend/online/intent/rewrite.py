# -*- coding: utf-8 -*-
"""
意图识别层：改写预留实现与跟进问句补全。
"""

import re

from backend.conf import REWRITE_PROMPT, STANDALONE_PROMPT, USE_REWRITE
from service.deepseek.http_client import HttpClient, ReplyKind

_FOLLOWUP_TOKENS = ("它", "它们", "这个", "那个", "这些", "那些", "刚才", "刚刚")


def _history_block(history: str) -> str:
    """历史非空时加上提示头。"""
    if not history:
        return ""
    return f"对话历史：\n{history}\n\n"


def _looks_like_followup(question: str) -> bool:
    """指代上一轮，或句首「那/还有」、句末「呢」，或含刚才/刚刚。完整问句返回 False。"""
    stripped = question.strip()
    stripped = re.sub(r"[。！？!?.,，.~～、]+$", "", stripped)
    if not stripped:
        return False
    if stripped.endswith("呢"):
        return True
    if stripped.startswith("那") or stripped.startswith("还有"):
        return True
    return any(token in stripped for token in _FOLLOWUP_TOKENS)


def _completed_text(prompt: str) -> str:
    """只要自然结束的非空正文。其它 kind 抛 RuntimeError，上层改用用户原文。"""
    reply = HttpClient().complete(prompt)
    if reply.kind == ReplyKind.STOP and reply.content.strip():
        return reply.content.strip()
    detail = reply.error_message or reply.content[:200]
    raise RuntimeError(f"{reply.kind} {detail}".strip())


def rewrite_query(text: str, history: str = "") -> str:
    """
    Query 改写预留实现：只替换 text。

    功能：开关打开则改成短检索问句；开关关闭时仅在有检索历史且问句像跟进时补全；否则原样。
    应用场景：在线图 intent 节点。
    输入：text — 查询原文；history — 仅检索轮的历史文本，可空。
    输出：检索用 text。
    """
    if USE_REWRITE:
        rewritten = _completed_text(
            REWRITE_PROMPT.format(
                history_block=_history_block(history),
                question=text,
            )
        )
        print(f"改写 {text!r} → {rewritten!r}")
        return rewritten
    if history and _looks_like_followup(text):
        rewritten = _completed_text(
            STANDALONE_PROMPT.format(
                history_block=_history_block(history),
                question=text,
            )
        )
        print(f"跟进补全 {text!r} → {rewritten!r}")
        return rewritten
    return text
