# -*- coding: utf-8 -*-
"""
会话约束：session 限长限字符；turns 窗口截断。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.conf import SESSION_ID_MAX_CHARS, SESSION_TURNS_MAX
from backend.online.graph import _clip_turns
from backend.online.schema import AskBody


def test_超长session被拒():
    """10 万字符那种 session 必须 422，不能进 MemorySaver。"""
    with pytest.raises(ValidationError) as excinfo:
        AskBody(text="你好", session="x" * (SESSION_ID_MAX_CHARS + 1))
    assert "session" in str(excinfo.value).lower() or "最长" in str(excinfo.value)


def test_非法session字符被拒():
    """路径分隔与空白不许进 session。"""
    with pytest.raises(ValidationError):
        AskBody(text="你好", session="../evil")
    with pytest.raises(ValidationError):
        AskBody(text="你好", session="a b")


def test_合法session放行():
    """前端常用的字母数字与连字符要过。"""
    body = AskBody(text="你好", session="demo-1_abc")
    assert body.session == "demo-1_abc"


def test_turns窗口只留最近N轮():
    """reducer 追加后长度不超过 SESSION_TURNS_MAX。"""
    existing = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(SESSION_TURNS_MAX)]
    clipped = _clip_turns(existing, [{"question": "new", "answer": "n"}])
    assert len(clipped) == SESSION_TURNS_MAX
    assert clipped[-1]["question"] == "new"
    assert clipped[0]["question"] == "q1"
