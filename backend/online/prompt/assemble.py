# -*- coding: utf-8 -*-
"""
提示词组装层。

使用前注意事项：出处不写进提示，由生成层用命中截片段做成 citations。
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from backend.conf import HANDBOOK_PROMPT as HANDBOOK_PROMPT_TEXT, SNIPPET_CHARS

HANDBOOK_PROMPT = ChatPromptTemplate.from_template(HANDBOOK_PROMPT_TEXT)


def _snippet(text: str) -> str:
    """命中正文截成引用片段。"""
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[:SNIPPET_CHARS] + "…"


def assemble_handbook_inputs(
    hits: list[dict],
    question: str,
    tags: list | None = None,
) -> tuple[dict, list[dict]]:
    """
    把检索命中收成提示变量，并收引用列表。

    功能：excerpts 与 identity 给 HANDBOOK_PROMPT；citations 由命中截原文片段，不让模型自编出处。
    应用场景：生成节点之前。
    输入：hits — 检索命中；question — 用户原文；tags — 本轮筛选芯片，可空。
    输出：({question, identity, excerpts}, citations)。
    """
    identity_tags = tags if tags is not None else []
    identity = "无"
    if identity_tags:
        identity = "；".join(f"{tag['field']}={tag['value']}" for tag in identity_tags)
    excerpts = []
    citations = []
    for index, hit in enumerate(hits, start=1):
        excerpts.append(f"[{index}] source={hit['source']} pages={hit['pages']}\n{hit['text']}")
        citations.append(
            {
                "id": hit["id"],
                "source": hit["source"],
                "pages": hit["pages"],
                "snippet": _snippet(hit["text"]),
            }
        )
    return {
        "question": question,
        "identity": identity,
        "excerpts": "\n\n".join(excerpts),
    }, citations
