# -*- coding: utf-8 -*-
"""
过滤层：来源强制筛选 + 过滤标签加权。

使用前注意事项：来源空字符串则不限来源。
标签**不做强制筛选**（ADR-0035）：它只在召回排序里加权——命中 N 条本轮标签的块加 N 分，
强制筛选会把「同字段多值」筛成 0 命中，直接把提问打成拒答。
标签在入库时从知识内容抽出；在线由意图识别对照库内已有对给出候选。
"""


def empty_filters() -> dict:
    """来源不限、标签对为空。"""
    return {"source": "", "tags": []}


def filters_are_set(filters: dict) -> bool:
    """来源或标签对任一非空，表示调用方要覆盖意图。"""
    if filters["source"]:
        return True
    return bool(filters["tags"])


def tags_from_filters(filters: dict) -> list[dict]:
    """
    把本轮筛选收成展示用标签列表。

    功能：来源非空则带 field=source；再追加 filters.tags。
    应用场景：生成结果的 tags；问答页芯片。
    输入：filters — source + tags[{field, value}]。
    输出：[{field, value}, ...]。
    """
    rows = []
    if filters["source"]:
        rows.append({"field": "source", "value": filters["source"]})
    for tag in filters["tags"]:
        rows.append({"field": tag["field"], "value": tag["value"]})
    return rows


def source_matches(source: str, filters: dict) -> bool:
    """
    来源维强制筛选。

    功能：来源是文档身份，相等才留；filters.source 为空串则该维不限。
    应用场景：召回前收窄候选（向量路与 BM25 路共用）。
    输入：source — 记录来源；filters — 只用 filters.source。
    输出：是否保留。
    """
    if filters["source"] and source != filters["source"]:
        return False
    return True


def tag_match_count(metadata: dict, wanted: list) -> int:
    """
    一个块命中了几条本轮标签。

    功能：逐条比对 (field, value)；旧记录缺 tags 当 []；重复的标签对只算一次。
    应用场景：召回排序加权（不是过滤）。
    输入：metadata — 含 tags；wanted — [{field, value}, ...]。
    输出：命中条数，0 表示不加权。
    """
    if not wanted:
        return 0
    record_tags = metadata["tags"] if "tags" in metadata else []
    available = {(tag["field"], tag["value"]) for tag in record_tags}
    unique = {(tag["field"], tag["value"]) for tag in wanted}
    return len(available & unique)
