# -*- coding: utf-8 -*-
"""
LLM 生成层：LangChain `prompt | model | parser` 与拒答。

使用前注意事项：密钥走 DEEPSEEK_API_KEY（进程环境变量或本仓 .env），不写进本仓。
"""

from __future__ import annotations

import json
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.conf import (
    CHAT_API_URL,
    CHAT_MODEL_NAME,
    CHAT_TIMEOUT_SEC,
    CHITCHAT_PROMPT,
    DEBUG_WRITE_QA,
    LLM_MAX_TOKENS,
    QA_DIR,
    REFUSE_MIN_SCORE,
    REFUSE_ON_LEXICAL_FALLBACK,
    REFUSE_TEXT,
)
from backend.offline.vector.store import tokenize
from backend.online.prompt.assemble import HANDBOOK_PROMPT, assemble_handbook_inputs

_PLAIN_PROMPT = ChatPromptTemplate.from_messages([("human", "{text}")])
_CHITCHAT_PROMPT = ChatPromptTemplate.from_template(CHITCHAT_PROMPT)
_PARSER = StrOutputParser()


def write_qa_result(name: str, payload) -> None:
    """
    把 JSON 写到 backend/data/result/qa/<name>。

    功能：DEBUG_WRITE_QA 关时直接返回，响应路径不碰磁盘。
    应用场景：拒答、闲聊、检索生成三条分支的调试题材写入磁盘。
    为什么默认关：每问覆盖写一次，写入磁盘抛 PermissionError 会穿透 router 变成 HTTP 500（TODO P1·20）。
    """
    if not DEBUG_WRITE_QA:
        return
    QA_DIR.mkdir(parents=True, exist_ok=True)
    path = QA_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {path}")


def _resolve_chat_api_key() -> str:
    """DEEPSEEK_API_KEY（进程环境变量或本仓 .env）。"""
    return os.getenv("DEEPSEEK_API_KEY", "").strip()


def _chat_base_url() -> str:
    """ChatOpenAI 要的是 /v1，不是完整 completions 路径。"""
    url = CHAT_API_URL.strip().rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[: -len(suffix)]
    return url


def _chat_model() -> ChatOpenAI:
    """按 conf 常量组一个 DeepSeek 兼容的 Chat 模型。"""
    api_key = _resolve_chat_api_key()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY：设进程环境变量或写入本仓 .env")
    return ChatOpenAI(
        model=CHAT_MODEL_NAME,
        api_key=api_key,
        base_url=_chat_base_url(),
        timeout=CHAT_TIMEOUT_SEC,
        max_tokens=LLM_MAX_TOKENS,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _invoke_text(prompt: ChatPromptTemplate, inputs: dict) -> str:
    """跑 prompt | model | parser，失败收成 RuntimeError。"""
    chain = prompt | _chat_model() | _PARSER
    try:
        return chain.invoke(inputs).strip()
    except Exception as exc:
        raise RuntimeError(f"Chat 失败: {type(exc).__name__}: {exc}") from exc


def chat_text(prompt: str) -> str:
    """
    调 Chat，取助手文本。

    功能：关思考模式；LCEL 产出字符串。
    应用场景：改写预留实现、意图分类、闲聊、入库抽标签。
    输入：prompt — user 正文。
    输出：助手文本（已 strip）。
    异常：缺 API key 或 Chat 失败 → RuntimeError。
    """
    return _invoke_text(_PLAIN_PROMPT, {"text": prompt})


def best_vector_score(hits: list[dict]):
    """
    命中里最好的向量余弦。

    功能：只读 vector_score；没有向量路的命中跳过。
    应用场景：拒答阈值；评测口打印。
    输入：hits — 检索命中。
    输出：最大余弦；没有任何向量分则 None。
    """
    values = []
    for hit in hits:
        if "vector_score" in hit:
            values.append(hit["vector_score"])
    if not values:
        return None
    return max(values)


def hits_contain_query_term(hits: list[dict], question: str) -> bool:
    """
    命中正文里是否真的出现了查询词。

    功能：按 tokenize 的词做判断（与 BM25 同一套切词判据），命中任意一条即算。
    应用场景：词面兜底时的拒答判据。
    输入：hits — 检索命中；question — 用户原文（或检索用问句）。
    输出：是否有词面重合。
    """
    query_tokens = set(tokenize(question))
    if not query_tokens:
        return False
    for hit in hits:
        if query_tokens & set(tokenize(hit.get("text", ""))):
            return True
    return False


def should_refuse(hits: list[dict], question: str = "", min_score: float | None = None) -> bool:
    """
    拒答预留实现：空命中，或最好的向量余弦低于 min_score；词面兜底另有一套判据。

    功能：阈值打在余弦上，不打在 RRF 融合分上。
          min_score 留空则用 conf.REFUSE_MIN_SCORE；`scripts/eval/eval_kb.py` 顶部 CALIBRATE=True 时靠它扫阈值，
          判据只保留这一份实现，不在评测口重抄一遍。
          向量路过的命中按余弦判；词面兜底的命中没有 vector_score，
          不能沿用「没有向量分就放行」——那等于在最该谨慎的时候关掉拒答开关，
          改为要求命中里出现查询词（REFUSE_ON_LEXICAL_FALLBACK）。
    应用场景：在线图 generate 节点；评测口负样本；阈值校准扫点。
    输入：hits — 检索命中；question — 查询原文，词面兜底时必传；min_score — 覆盖阈值，≤0 视为关掉余弦判据。
    输出：是否拒答。
    """
    if not hits:
        return True
    if REFUSE_ON_LEXICAL_FALLBACK and any(hit.get("lexical_fallback") for hit in hits):
        # 词面兜底：没有词面重合就不算有证据
        return not hits_contain_query_term(hits, question)
    threshold = REFUSE_MIN_SCORE if min_score is None else min_score
    if threshold <= 0:
        return False
    cosine = best_vector_score(hits)
    if cosine is None:
        return False
    return cosine < threshold


def refuse_result(question: str, debug_name: str = "step3_answer.json") -> dict:
    """
    证据不足时的生成结果。

    功能：固定拒答文案，citations 为空列表。
    应用场景：在线图拒答分支。
    输入：question — 用户原问；debug_name — 调试题材文件名（调用方按 session 组装，TODO P2·31）。
    输出：{answer, citations}。
    """
    result = {"answer": REFUSE_TEXT, "citations": []}
    write_qa_result(debug_name, {"question": question, **result})
    print("[阶段3] 拒答")
    return result


def step3_chitchat(question: str, debug_name: str = "step3_answer.json") -> dict:
    """
    闲聊生成：不检索，引用为空。

    功能：CHITCHAT_PROMPT | model | parser；不是拒答。
    应用场景：在线图 route=chitchat 时跳过召回后的生成。
    输入：question — 用户原文；debug_name — 调试题材文件名。
    输出：{answer, citations}，citations 为 []。
    异常：缺 API key 或 Chat 失败 → RuntimeError。
    """
    print("[阶段3] 闲聊")
    answer = _invoke_text(_CHITCHAT_PROMPT, {"question": question})
    result = {"answer": answer, "citations": []}
    write_qa_result(debug_name, {"question": question, "route": "chitchat", **result})
    print(f"问 → {question}")
    print(f"答 → {answer}")
    return result


def step3_generate(
    hits: list[dict],
    question: str,
    tags: list | None = None,
    debug_name: str = "step3_answer.json",
) -> dict:
    """
    按命中生成答案并附引用。

    功能：组装变量后 HANDBOOK_PROMPT | model | parser；citations 由命中截片段。
    应用场景：在线图生成节点。
    输入：hits — 召回结果；question — 用户原文；tags — 本轮筛选芯片，可空；
          debug_name — 调试题材文件名。
    输出：生成结果 {answer, citations}。
    异常：缺 API key 或 Chat 失败 → RuntimeError。
    """
    print("[阶段3] 生成")
    inputs, citations = assemble_handbook_inputs(hits, question, tags)
    answer = _invoke_text(HANDBOOK_PROMPT, inputs)
    result = {"answer": answer, "citations": citations}
    write_qa_result(debug_name, {"question": question, **result})
    print(f"问 → {question}")
    print(f"答 → {answer}")
    for citation in citations:
        print(f"  引用 {citation['id']} {citation['source']} p.{citation['pages']}: {citation['snippet']}")
    return result
