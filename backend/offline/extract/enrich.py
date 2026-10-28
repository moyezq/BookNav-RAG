# -*- coding: utf-8 -*-
"""
内容生成信息：从知识正文抽出过滤标签。

使用前注意事项：来源仍是文档身份，不写入 tags。字段名随文档变，不预置品牌/车型。抽不出则为 []，
但**要说出来**（哪一步抽空的、模型原始输出前 200 字），别让「没有标签的文档」混在「入库成功」里。
抠 JSON 用花括号配对逐段试解析，不用贪婪正则（多对象输出会被整段框住 → 解析必失败）。
"""

from __future__ import annotations

import json

from backend.online.generate.answer import chat_text

INGEST_TAG_CHARS = 4000
INGEST_TAG_PAGES = 8
INGEST_TAG_MAX = 8
INGEST_TAG_PROMPT = (
    "你从文档正文抽出便于过滤检索的标签。只输出 JSON，不要解释。\n"
    '格式：{{"tags":[{{"field":"字段名","value":"值"}}]}}\n'
    "字段名用简短中文或英文，随文档领域而定，不要假定是汽车或任何行业。\n"
    "没有依据则 tags 为 []。最多 {max_tags} 条。不要重复。\n"
    "不要把文档文件名或来源身份当成标签。\n"
    "来源身份（不要写入 tags）：{source}\n\n"
    "正文片段：\n{sample}"
)


def sample_pages_text(page_rows: list[dict], max_pages: int = INGEST_TAG_PAGES) -> str:
    """
    取前若干页拼成抽标签用的正文原文片段。

    功能：按页顺序拼接，截到 INGEST_TAG_CHARS。
    应用场景：可分块原文写入前。
    输入：page_rows — [{page, text}, ...]；max_pages — 最多取几页。
    输出：原文片段字符串；没有正文则空串。
    """
    parts = []
    for page in page_rows[:max_pages]:
        text = page["text"] if "text" in page and page["text"] else ""
        if text:
            parts.append(text)
    sample = "\n".join(parts)
    if len(sample) > INGEST_TAG_CHARS:
        return sample[:INGEST_TAG_CHARS]
    return sample


def sample_content_text(content: dict) -> str:
    """
    从文件内容收成抽标签用的原文片段。

    功能：有 pages 走 sample_pages_text；否则截 text。
    应用场景：collect 之前。
    输入：content — 抽取方式产出。
    输出：原文片段字符串。
    """
    if "pages" in content:
        return sample_pages_text(content["pages"])
    text = content["text"] if "text" in content and content["text"] else ""
    if len(text) > INGEST_TAG_CHARS:
        return text[:INGEST_TAG_CHARS]
    return text


def _iter_json_object_candidates(raw: str) -> list[str]:
    """
    从模型输出里抠出所有「花括号配对」的候选 JSON 对象。

    功能：逐字符扫，遇到 `{` 记起点并累深度，回到深度 0 就是一段候选；字符串里的花括号不计数。
    应用场景：_parse_tag_json 用来顶掉贪婪正则 `\\{.*\\}`——那个在多对象输出上会跨对象炸掉。
    输入：raw — 模型原始输出。
    输出：候选子串列表，按出现顺序；没有则 []。
    """
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


def _parse_tag_json(raw: str, source_name: str) -> list[dict]:
    """
    从模型输出里取出 tags 列表。

    功能：逐段候选 JSON 试解析，取**第一个**含非空 tags 列表的对象；来源身份、空项、重复项丢掉；
          最多 INGEST_TAG_MAX 条；全都解析不出来则返回 []（不抛）。
    应用场景：infer_ingest_tags 解析模型输出。
    输入：raw — 模型原始输出（可能有「思考」文字、代码围栏或第二个 JSON）；source_name — 文档身份。
    输出：[{field, value}, ...]；认不出则 []。
    为什么不用贪婪正则（TODO P1·12）：`re.search(r"\\{.*\\}", raw, DOTALL)` 在多对象输出上从第一个 `{`
      一直框到最后一个 `}`，json.loads 必失败 → 整个 tags 变 [] 而入库照样算成功。
    """
    payload = None
    for candidate in _iter_json_object_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("tags"), list):
            payload = parsed
            break
    if payload is None:
        return []
    rows = []
    seen = set()
    for item in payload["tags"]:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        value = str(item.get("value") or "").strip()
        if not field or not value:
            continue
        if field == "source" or value == source_name:
            continue
        key = (field, value)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"field": field, "value": value})
        if len(rows) >= INGEST_TAG_MAX:
            break
    return rows


def infer_ingest_tags(sample_text: str, source_name: str) -> list[dict]:
    """
    从知识正文抽出过滤标签。

    功能：调 Chat；不要来源身份；失败或解析不出则 []，并且**把原因打出来**（抽空不是静默降级）。
    应用场景：写可分块原文时写入 tags。
    输入：sample_text — 正文原文片段；source_name — 文档身份。
    输出：[{field, value}, ...]，抽不出则为 []。
    """
    sample = (sample_text or "").strip()
    if not sample:
        print("入库标签：正文为空，tags=[]（这份来源不会按标签加权）")
        return []
    try:
        raw = chat_text(
            INGEST_TAG_PROMPT.format(
                max_tags=INGEST_TAG_MAX,
                source=source_name,
                sample=sample,
            )
        )
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"入库标签抽取失败，tags=[]（这份来源不会按标签加权）：{type(exc).__name__}: {exc}")
        return []
    tags = _parse_tag_json(raw, source_name)
    if not tags:
        # 解析不出来时把模型输出前 200 字打出来：不然「入库成功 + 没有标签」看不出是哪一步坏的
        print(
            f"入库标签解析不出 tags，tags=[]（这份来源不会按标签加权）："
            f"原始输出 {len(raw or '')} 字，前 200 字 {((raw or '')[:200])!r}"
        )
        return []
    print(f"入库标签 tags={len(tags)} 条：{tags}")
    return tags
