# -*- coding: utf-8 -*-
"""
问答 HTTP 的请求与响应 schema。

职责：POST /api/ask 的 Pydantic 模型。
对外：AskBody、GenerationResult 等；字段与 frontend/src/api.ts 镜像。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from backend.conf import SESSION_ID_MAX_CHARS

_SESSION_OK = re.compile(r"^[A-Za-z0-9_-]*$")


class FilterTag(BaseModel):
    """一条筛选标签：来源或入库抽出的键值。"""

    field: str
    value: str

    @field_validator("field", "value")
    @classmethod
    def _strip(cls, value: str) -> str:
        """去掉首尾空白：筛选值要与库内标签逐字相等。"""
        return value.strip()


class AskFilters(BaseModel):
    """查询约定上的可选过滤。source 强制筛选；tags 空则不加权（ADR-0035）。"""

    source: str = ""
    tags: list[FilterTag] = Field(default_factory=list)

    @field_validator("source")
    @classmethod
    def _strip(cls, value: str) -> str:
        """来源要与库内 source 逐字相等，先去空白。"""
        return value.strip()


class AskBody(BaseModel):
    """POST /api/ask 请求体。"""

    text: str
    session: str = ""
    filters: AskFilters = Field(default_factory=AskFilters)

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        """问句去首尾空白；空串仍由路由层判 400。"""
        return value.strip()

    @field_validator("session")
    @classmethod
    def _session(cls, value: str) -> str:
        """会话 id：去空白；限长；只留字母数字与 _ -（与调试文件名清洗一致）。"""
        cleaned = (value or "").strip()
        if len(cleaned) > SESSION_ID_MAX_CHARS:
            raise ValueError(f"session 最长 {SESSION_ID_MAX_CHARS} 个字符")
        if cleaned and _SESSION_OK.fullmatch(cleaned) is None:
            raise ValueError("session 只允许字母、数字、下划线、连字符")
        return cleaned


class Citation(BaseModel):
    """生成结果里的一条引用。"""

    id: str
    source: str
    pages: list[int]
    snippet: str


class GenerationResult(BaseModel):
    """POST /api/ask 成功响应。"""

    answer: str
    citations: list[Citation]
    tags: list[FilterTag]
