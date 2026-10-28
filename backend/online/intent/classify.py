# -*- coding: utf-8 -*-
"""
意图识别：闲聊/检索分类，并对照库内过滤标签给出本轮加权提示（ADR-0035：标签不强制筛选）。标签候选在本文件，不是过滤层。
"""

from __future__ import annotations

import json
import re

from backend.online.filter.tags import empty_filters
from backend.online.generate.answer import chat_text
from backend.conf import CLASSIFY_PROMPT, HISTORY_MAX_TURNS, QUERY_TAG_PROMPT, TAG_LEXICAL_MIN_CHARS

ROUTE_RETRIEVE = "retrieve"
ROUTE_CHITCHAT = "chitchat"

_CHITCHAT_EXACT = frozenset(
    {
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "hello",
        "hi",
        "hey",
        "谢谢",
        "谢谢你",
        "谢谢您",
        "感谢",
        "多谢",
        "再见",
        "拜拜",
        "bye",
        "goodbye",
        "你是谁",
        "你是什么",
        "你叫什么",
        "在吗",
        "早上好",
        "晚上好",
        "下午好",
    }
)
_IDENTITY_MARKERS = (
    "哪个品牌",
    "哪个车",
    "哪款车",
    "什么品牌",
    "什么车型",
    "哪份",
    "资料来源",
    "来源于",
    "来自哪",
)


def format_turns(
    turns: list[dict],
    limit: int = HISTORY_MAX_TURNS,
    routes: set[str] | None = None,
) -> str:
    """
    把最近若干轮收成意图用的历史文本。

    功能：每轮用户问、助手答；有 tags 再加筛选行。空列表得空串。routes 非空则只收这些路由的轮次。
    应用场景：分类用全历史；改写/标签只用检索轮。
    输入：turns — [{question, answer, route}, ...]，可带 tags；limit — 最多取末尾几轮；routes — 可空。
    输出：拼接文本，无轮次时为 ""。
    """
    if not turns:
        return ""
    selected = turns
    if routes is not None:
        selected = [turn for turn in turns if turn.get("route") in routes]
    if not selected:
        return ""
    recent = selected[-limit:]
    lines: list[str] = []
    for turn in recent:
        lines.append(f"用户：{turn['question']}")
        lines.append(f"助手：{turn['answer']}")
        if "tags" in turn and turn["tags"]:
            parts = [f"{tag['field']}={tag['value']}" for tag in turn["tags"]]
            lines.append("筛选：" + "；".join(parts))
    return "\n".join(lines)


def _history_block(history: str) -> str:
    """历史非空时加上提示头，供 prompt.format。"""
    if not history:
        return ""
    return f"对话历史：\n{history}\n\n"


def _normalize_question(question: str) -> str:
    """去掉首尾空白和末尾标点，便于整句比对。"""
    stripped = question.strip()
    stripped = re.sub(r"[。！？!?.,，.~～、]+$", "", stripped)
    return stripped.casefold()


def is_identity_question(question: str) -> bool:
    """
    本轮是否在问刚才答案对应的文档身份。

    功能：命中来源/品牌/车型/哪份文档的问法。
    应用场景：有检索历史时强制 retrieve；沿用上一轮筛选。
    输入：question — 用户原文。
    输出：是否身份追问。
    """
    return any(marker in question for marker in _IDENTITY_MARKERS)


def _lexical_route(question: str) -> str:
    """整句命中闲聊套话则 chitchat，否则空串交给模型。"""
    if _normalize_question(question) in _CHITCHAT_EXACT:
        return ROUTE_CHITCHAT
    return ""


def _parse_route_json(raw: str) -> str:
    """从模型输出取出 route；认不出则 retrieve。"""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return ROUTE_RETRIEVE
    payload = json.loads(match.group(0))
    route = str(payload.get("route", "")).strip()
    if route == ROUTE_CHITCHAT:
        return ROUTE_CHITCHAT
    return ROUTE_RETRIEVE


def classify_route(question: str, history: str = "") -> str:
    """
    判断本轮走检索还是闲聊。

    功能：套话整句匹配；有历史且问文档身份则 retrieve；否则模型分类。失败或拿不准都 retrieve。
    应用场景：在线图 intent 节点，写入 route。
    输入：question — 用户原文；history — format_turns 文本，可空。
    输出：retrieve 或 chitchat。
    """
    lexical = _lexical_route(question)
    if lexical:
        print(f"意图分类 套话 {lexical}")
        return lexical
    if history and is_identity_question(question):
        print("意图分类 追问文档身份 retrieve")
        return ROUTE_RETRIEVE
    try:
        raw = chat_text(
            CLASSIFY_PROMPT.format(
                history_block=_history_block(history),
                question=question,
            )
        )
        route = _parse_route_json(raw)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"意图分类失败，按检索: {type(exc).__name__}: {exc}")
        return ROUTE_RETRIEVE
    print(f"意图分类 模型 {route}")
    return route


def _longest_catalog_hit(question: str, values: list[str]) -> str:
    """问句里出现的目录值；更长的优先，避免短串误伤。"""
    ranked = sorted((value for value in values if value), key=len, reverse=True)
    for value in ranked:
        if value in question:
            return value
    return ""


def _parse_filter_json(raw: str) -> dict:
    """从模型输出取出 source 与 tags。认不出则全空。"""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return empty_filters()
    payload = json.loads(match.group(0))
    filters = empty_filters()
    if "source" in payload and payload["source"]:
        filters["source"] = str(payload["source"]).strip()
    if "tags" in payload and payload["tags"]:
        rows = []
        for item in payload["tags"]:
            field = item["field"].strip() if "field" in item and item["field"] else ""
            value = item["value"].strip() if "value" in item and item["value"] else ""
            if field and value and field != "source":
                rows.append({"field": field, "value": value})
        filters["tags"] = rows
    return filters


def _constrain_to_catalog(filters: dict, catalog: dict) -> dict:
    """只保留目录里有的来源与标签对。"""
    constrained = empty_filters()
    if filters["source"] and filters["source"] in catalog["sources"]:
        constrained["source"] = filters["source"]
    allowed = {(tag["field"], tag["value"]) for tag in catalog["tags"]}
    rows = []
    for tag in filters["tags"]:
        if (tag["field"], tag["value"]) in allowed:
            rows.append({"field": tag["field"], "value": tag["value"]})
    constrained["tags"] = rows
    return constrained


def _lexical_filters(question: str, catalog: dict) -> dict:
    """
    问句直接点名的来源或标签值（代码做的确定性那一路）。

    功能：来源取最长的目录命中；标签值取问句里字面出现的全部对——**同字段多值都收**，
          它们只参与加权，命中越多分越高，不像以前那样被 AND 切成 0 命中（ADR-0035）。
          比较时忽略空格：库里的值是「问界 新M5 增程版」，用户说的是「问界新M5」。
          短于 TAG_LEXICAL_MIN_CHARS 的值不参与，避免单字误伤。
    应用场景：infer_query_filters 的第一路。
    输入：question — 用户原文；catalog — {sources, tags}。
    输出：{source, tags}。
    """
    filters = empty_filters()
    filters["source"] = _longest_catalog_hit(question, catalog["sources"])
    squeezed = re.sub(r"\s+", "", question)
    ranked = sorted(catalog["tags"], key=lambda tag: len(tag["value"]), reverse=True)
    rows = []
    seen = set()
    for tag in ranked:
        value = tag["value"]
        if len(value) < TAG_LEXICAL_MIN_CHARS or re.sub(r"\s+", "", value) not in squeezed:
            continue
        key = (tag["field"], value)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"field": tag["field"], "value": value})
    filters["tags"] = rows
    return filters


def infer_query_filters(question: str, catalog: dict, history: str = "") -> dict:
    """
    按本轮问句从库内标签目录选出加权用的过滤值。

    功能：先做目录子串匹配（代码那一路）；**字面命中就不调模型**（省一次 LLM）；
          都没命中才让模型从目录里挑，补字面命中不到的同义说法。
          选出的 tags 交给召回层加权，不再当强制筛选（ADR-0035）。
    应用场景：在线图 intent 节点，写入本轮 filters。
    输入：question — 用户原文；catalog — {sources, tags}；history — 可空。
    输出：{source, tags}。source 空串=不限来源；tags 空=不加权。
    """
    lexical = _lexical_filters(question, catalog)
    if lexical["source"] or lexical["tags"]:
        print(f"意图标签 子串 {lexical}")
        return lexical
    has_catalog = bool(catalog["sources"] or catalog["tags"])
    if not has_catalog:
        return empty_filters()
    pair_lines = [f"{tag['field']}={tag['value']}" for tag in catalog["tags"]]
    catalog_text = "来源: " + ", ".join(catalog["sources"]) + "\n标签: " + "; ".join(pair_lines)
    try:
        raw = chat_text(
            QUERY_TAG_PROMPT.format(
                catalog=catalog_text,
                history_block=_history_block(history),
                question=question,
            )
        )
        inferred = _constrain_to_catalog(_parse_filter_json(raw), catalog)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        print(f"意图标签抽取失败，不过滤: {type(exc).__name__}: {exc}")
        return empty_filters()
    print(f"意图标签 模型 {inferred}")
    return inferred
