# -*- coding: utf-8 -*-
"""
在线主路径：LangGraph 图（意图后条件边）。

职责：改写 / 分类 / 标签 → 检索或跳过 → rerank → 拒答、闲聊或生成。
步骤：intent →（retrieve）recall → rerank → generate；闲聊 intent → generate
    重排 截 TOP_K 后再并同节邻块。
对外：build_ask_graph、run_ask_graph
前置：知识库已 load；预留开关在 backend/conf/base.py。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.conf import DEBUG_TAG_MAX_CHARS, DEBUG_WRITE_QA, SESSION_TURNS_MAX
from backend.online.filter.tags import empty_filters, filters_are_set, tags_from_filters
from backend.online.generate.answer import refuse_result, should_refuse, step3_chitchat, step3_generate
from backend.online.intent.classify import (
    ROUTE_CHITCHAT,
    ROUTE_RETRIEVE,
    classify_route,
    format_turns,
    infer_query_filters,
    is_identity_question,
)
from backend.online.intent.rewrite import _looks_like_followup, rewrite_query
from backend.online.recall.retrieve import KnowledgeBase, retrieve_hits
from backend.online.rerank.overlap import rerank_hits

# 进程内会话快照；无 session 时每次新 thread，互不串历史
_checkpointer = MemorySaver()
_compiled = None
_compiled_kb_id = None


def debug_name(base: str, session: str) -> str:
    """
    调试题材的文件名：基名 + 清洗后的 session。

    功能：文件名 = `<stem>_<session>.<后缀>`；session 只留 [A-Za-z0-9_-] 且截到
          DEBUG_TAG_MAX_CHARS——它由客户端给，直接拼进文件名会写出 qa 目录之外
          （`../../x`、路径分隔符）。一个合法字符都没留下时退回「毫秒时间戳 + 4 位随机」，
          于是没 session 的并发两问也各写各的，不再最后写者赢。
    应用场景：recall 与 generate 两个节点组装调试题材名；DEBUG_WRITE_QA 关着时没人用它。
    输入：base — 基名（如 step2_hits.json）；session — 会话标识，可空。
    输出：文件名（不含目录）。
    """
    stem, _, suffix = base.rpartition(".")
    if not stem:
        stem, suffix = base, "json"
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session or "")[:DEBUG_TAG_MAX_CHARS]
    if not safe:
        safe = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}-{uuid4().hex[:4]}"
    return f"{stem}_{safe}.{suffix}"


def _clip_turns(existing: list, new: list) -> list:
    """追加本轮 turns，只保留最近 SESSION_TURNS_MAX 轮，避免 MemorySaver 只增不减。"""
    merged = list(existing or []) + list(new or [])
    if len(merged) <= SESSION_TURNS_MAX:
        return merged
    return merged[-SESSION_TURNS_MAX:]


class AskState(TypedDict):
    """在线图状态：查询约定字段加中间命中、路由与会话轮次。"""

    question: str
    text: str
    session: str
    filters: dict | None
    hits: list
    answer: str
    citations: list
    tags: list
    route: str
    rewrite_failed: bool
    turns: Annotated[list, _clip_turns]


def _last_retrieve_turn(turns: list) -> dict | None:
    """最近一条检索轮；没有则 None。"""
    for turn in reversed(turns):
        if turn["route"] == ROUTE_RETRIEVE:
            return turn
    return None


def _normalize_filters(raw) -> dict:
    """
    把调用方给的过滤收成 {source, tags}。

    功能：认 dict 与带属性的对象（HTTP 层是 Pydantic 模型）；空项丢掉，source 与 tag 都去首尾空白。
    应用场景：图入口与检索核心。
    输入：raw — dict、对象或 None。
    输出：{source: str, tags: [{field, value}, ...]}；没有则空过滤。
    """
    filters = empty_filters()
    if raw is None:
        return filters
    if isinstance(raw, dict):
        source = raw.get("source", "")
        tags = raw.get("tags", [])
    else:
        source = getattr(raw, "source", "")
        tags = getattr(raw, "tags", []) or []
    # 只留空白等于没点名来源；不 strip 会让 "  " 逐字比库内 source 而静默 0 命中
    source = str(source).strip() if source else ""
    if source:
        filters["source"] = source
    rows = []
    for tag in tags or []:
        if isinstance(tag, dict):
            field = tag.get("field", "")
            value = tag.get("value", "")
        else:
            field = getattr(tag, "field", "")
            value = getattr(tag, "value", "")
        field = str(field).strip() if field else ""
        value = str(value).strip() if value else ""
        if field and value:
            rows.append({"field": field, "value": value})
    filters["tags"] = rows
    return filters


def _build_rewrite(question: str, retrieve_history: str) -> tuple[str, bool]:
    """
    跑改写预留实现；失败退化为用户原文。

    功能：改写是可关的辅助调用，它的失败不允许打死整次提问（否则 HTTP 层会 502）。
    应用场景：图 intent 节点；评测核心。
    输入：question — 用户原文；retrieve_history — 仅检索轮的历史文本，可空。
    输出：(检索用问句, 是否发生降级)。
    """
    try:
        text = rewrite_query(question, retrieve_history)
    except RuntimeError as exc:
        print(f"改写失败，改用用户原文检索: {type(exc).__name__}: {exc}")
        return question, True
    return text, False


def route_after_intent(state: AskState) -> Literal["recall", "generate"]:
    """闲聊跳过召回，直接生成。"""
    if state["route"] == ROUTE_CHITCHAT:
        return "generate"
    return "recall"


def build_ask_graph(knowledge_base: KnowledgeBase):
    """
    编译一条带条件边的问答图。

    功能：intent 写入 route；闲聊不召回；生成后把本轮追加进 turns。
    应用场景：HTTP 与命令行共用。
    输入：knowledge_base — 已 load 的知识库。
    输出：compile 后的图。
    """

    def intent(state: AskState) -> dict:
        """意图识别：读历史，分类，改写，给出本轮筛选。"""
        history = format_turns(state["turns"])
        retrieve_history = format_turns(state["turns"], routes={ROUTE_RETRIEVE})
        normalized = _normalize_filters(state["filters"])
        question = state["question"]
        last = _last_retrieve_turn(state["turns"])
        previous = _normalize_filters(last["filters"]) if last is not None and "filters" in last else empty_filters()
        reuse_last = (
            last is not None
            and filters_are_set(previous)
            and (_looks_like_followup(question) or is_identity_question(question))
        )
        # 调试覆盖标签时强制检索，不走闲聊
        if filters_are_set(normalized):
            route = ROUTE_RETRIEVE
            text, rewrite_failed = _build_rewrite(question, retrieve_history)
            filters = normalized
            print(f"意图标签 调用方覆盖 {filters}")
        else:
            route = classify_route(question, history)
            if route == ROUTE_CHITCHAT:
                return {
                    "text": question,
                    "filters": empty_filters(),
                    "tags": [],
                    "route": route,
                    "hits": [],
                    "rewrite_failed": False,
                }
            if reuse_last and is_identity_question(question) and "text" in last and last["text"]:
                filters = previous
                text = last["text"]
                rewrite_failed = False
                print(f"意图标签 沿用上一轮 {filters}")
                print(f"身份追问 检索沿用 {text!r}")
            else:
                if reuse_last:
                    filters = previous
                    print(f"意图标签 沿用上一轮 {filters}")
                else:
                    filters = infer_query_filters(
                        question,
                        knowledge_base.tag_catalog(),
                        retrieve_history,
                    )
                text, rewrite_failed = _build_rewrite(question, retrieve_history)
        return {
            "text": text,
            "filters": filters,
            "tags": tags_from_filters(filters),
            "route": route,
            "rewrite_failed": rewrite_failed,
        }

    def recall(state: AskState) -> dict:
        """召回节点：来源强制筛选与标签加权都在 search 内完成。"""
        hits = retrieve_hits(
            knowledge_base,
            state["text"],
            state["filters"],
            debug_name=debug_name("step2_hits.json", state.get("session", "")),
        )
        return {"hits": hits}

    def rerank(state: AskState) -> dict:
        """重排节点：截 TOP_K，再并同节邻块。"""
        cut = rerank_hits(state["hits"])
        return {"hits": knowledge_base.expand_hits_by_catalogue(cut)}

    def generate(state: AskState) -> dict:
        """生成节点：闲聊、拒答或按命中生成；本轮写入 turns。"""
        debug_file = debug_name("step3_answer.json", state.get("session", ""))
        if state["route"] == ROUTE_CHITCHAT:
            result = step3_chitchat(state["question"], debug_name=debug_file)
        elif should_refuse(state["hits"], state["question"]):
            result = refuse_result(state["question"], debug_name=debug_file)
        else:
            result = step3_generate(
                state["hits"], state["question"], state["tags"], debug_name=debug_file
            )
        turn = {
            "question": state["question"],
            "text": state["text"],
            "answer": result["answer"],
            "route": state["route"],
            "tags": state["tags"],
            "filters": state["filters"] or empty_filters(),
        }
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "turns": [turn],
        }

    graph = (
        StateGraph(AskState)
        .add_node("intent", intent)
        .add_node("recall", recall)
        .add_node("rerank", rerank)
        .add_node("generate", generate)
        .add_edge(START, "intent")
        .add_conditional_edges(
            "intent",
            route_after_intent,
            {"recall": "recall", "generate": "generate"},
        )
        .add_edge("recall", "rerank")
        .add_edge("rerank", "generate")
        .add_edge("generate", END)
        .compile(checkpointer=_checkpointer)
    )
    return graph


def _ask_graph(knowledge_base: KnowledgeBase):
    """同一知识库对象复用已编译图，保住 MemorySaver 会话。"""
    global _compiled, _compiled_kb_id
    kb_id = id(knowledge_base)
    if _compiled is None or _compiled_kb_id != kb_id:
        _compiled = build_ask_graph(knowledge_base)
        _compiled_kb_id = kb_id
    return _compiled


def _empty_turns() -> list:
    """空历史。"""
    return []


def ask_once_core(
    knowledge_base: KnowledgeBase,
    question: str,
    turns: list | None = None,
    filters: dict | None = None,
    route_override: str = "",
    write_debug: bool = DEBUG_WRITE_QA,
) -> dict:
    """
    单轮检索：与线上图同一顺序，停在重排之后，不调 LLM 生成。

    功能：分类 → 改写 → 本轮筛选（沿用上一轮 / 对照库内目录）→ 召回 → 重排截 TOP_K → 同节邻块。
          步骤与顺序跟图里的 intent / recall / rerank 一致；这是评测能对齐线上路径的前提。
    应用场景：scripts/eval/eval_kb.py 正负样本判定；pytest 的在线路径测试。
    输入：knowledge_base — 已 load；question — 用户原文；turns — 历史轮（与图 state 同形状），可空；
          filters — 调用方覆盖，非空则强制检索；route_override — 注入路由，验收用（传 chitchat 时不该召回）；
          write_debug — 是否写 backend/data/result/qa 调试题材，默认取 conf 的 DEBUG_WRITE_QA（默认关）；评测与测试传 False。
    输出：{route, text, filters, hits, turns, rewrite_failed}；route=chitchat 时 hits 为空。
    """
    history = format_turns(turns or _empty_turns())
    retrieve_history = format_turns(turns or _empty_turns(), routes={ROUTE_RETRIEVE})
    normalized = _normalize_filters(filters)
    last = _last_retrieve_turn(turns or _empty_turns())
    previous = _normalize_filters(last["filters"]) if last is not None and "filters" in last else empty_filters()
    reuse_last = (
        last is not None
        and filters_are_set(previous)
        and (_looks_like_followup(question) or is_identity_question(question))
    )

    # 意图识别：顺序与图 intent 节点一致
    if route_override:
        route = route_override
    elif filters_are_set(normalized):
        route = ROUTE_RETRIEVE
    else:
        route = classify_route(question, history)

    if route == ROUTE_CHITCHAT:
        return {
            "route": route,
            "text": question,
            "filters": empty_filters(),
            "hits": [],
            "turns": turns or [],
            "rewrite_failed": False,
        }

    if filters_are_set(normalized):
        resolved = normalized
        text, rewrite_failed = _build_rewrite(question, retrieve_history)
    elif reuse_last and is_identity_question(question) and "text" in last and last["text"]:
        resolved = previous
        text = last["text"]
        rewrite_failed = False
    else:
        if reuse_last:
            resolved = previous
        else:
            resolved = infer_query_filters(question, knowledge_base.tag_catalog(), retrieve_history)
        text, rewrite_failed = _build_rewrite(question, retrieve_history)

    # 召回 → 重排（截 TOP_K）→ 同节邻块
    hits = retrieve_hits(knowledge_base, text, resolved, write_debug=write_debug)
    cut = rerank_hits(hits)
    expanded = knowledge_base.expand_hits_by_catalogue(cut)
    return {
        "route": route,
        "text": text,
        "filters": resolved,
        "hits": expanded,
        "turns": turns or [],
        "rewrite_failed": rewrite_failed,
    }


def run_ask_graph(
    knowledge_base: KnowledgeBase,
    question: str,
    filters: dict | None = None,
    session: str = "",
) -> dict:
    """
    跑在线主路径，打出生成结果。

    功能：invoke 图；session 非空则按会话续 MemorySaver，turns 只给意图读。
    应用场景：python -m backend.online.server；scripts/ask/query_kb.py；scripts/ask/chat_kb.py。
    输入：knowledge_base — step1；question — 用户原文；filters — 可空，非空则覆盖意图标签；session — 会话 id。
    输出：生成结果 {answer, citations, tags}；另带 route、text 给命令行，HTTP schema 不收这两项。
    """
    thread_id = session.strip() if session.strip() else str(uuid4())
    initial: AskState = {
        "question": question,
        "text": question,
        "session": thread_id,
        "filters": filters,
        "hits": [],
        "answer": "",
        "citations": [],
        "tags": [],
        "route": ROUTE_RETRIEVE,
        "rewrite_failed": False,
        "turns": [],
    }
    final = _ask_graph(knowledge_base).invoke(
        initial,
        {"configurable": {"thread_id": thread_id}},
    )
    return {
        "answer": final["answer"],
        "citations": final["citations"],
        "tags": final["tags"],
        "route": final["route"],
        "text": final["text"],
    }
