# -*- coding: utf-8 -*-
"""
过滤层的单元测试。

职责：锁定 backend/online/filter/tags.py 的含义——
      `source_matches`：空串不限来源、来源精确相等（按来源强制筛选，ADR-0035 之后只留来源这一维）；
      `tag_match_count`：标签只算命中条数（0 = 不加权），缺 tags 的旧块当 []，
      同字段多值各算一条（不再 AND 成 0）。

应用场景：改过滤逻辑之后跑 python -m pytest tests/test_filters.py。
"""

from __future__ import annotations

from backend.online.filter.tags import (
    empty_filters,
    filters_are_set,
    source_matches,
    tag_match_count,
    tags_from_filters,
)


def _metadata(**tags):
    """按 field=value 造一份块 metadata。"""
    return {"tags": [{"field": field, "value": value} for field, value in tags.items()]}


def test_空来源不限():
    """source 空串 = 该维不限，标签不参与强制筛选。"""
    assert source_matches("某手册", empty_filters()) is True


def test_来源精确相等且不做子串匹配():
    """source 用相等判断：手册AB 不是手册A。"""
    filters = empty_filters()
    filters["source"] = "手册A"

    assert source_matches("手册A", filters) is True
    assert source_matches("手册AB", filters) is False


def test_标签不再把块筛掉():
    """ADR-0035：带标签的 filters 也不许丢记录——只影响分数。"""
    filters = {"source": "", "tags": [{"field": "品牌", "value": "比亚迪"}]}

    assert source_matches("某手册", filters) is True


def test_标签命中按条数计数():
    """命中几条算几条。"""
    metadata = _metadata(品牌="比亚迪", 车型="汉EV")

    assert tag_match_count(metadata, []) == 0
    assert tag_match_count(metadata, [{"field": "品牌", "value": "比亚迪"}]) == 1
    assert (
        tag_match_count(
            metadata,
            [{"field": "品牌", "value": "比亚迪"}, {"field": "车型", "value": "汉EV"}],
        )
        == 2
    )


def test_同名不同值不算命中():
    """字段名对、值不对，不算命中。"""
    metadata = _metadata(品牌="比亚迪")

    assert tag_match_count(metadata, [{"field": "品牌", "value": "特斯拉"}]) == 0
    assert tag_match_count(metadata, [{"field": "车型", "value": "比亚迪"}]) == 0


def test_同字段多值各算一条():
    """同字段两个值过去走 AND 必然 0 命中，现在各算一条加权。"""
    metadata = _metadata(主题="安全", 品牌="比亚迪")

    matched = tag_match_count(
        metadata,
        [{"field": "主题", "value": "安全"}, {"field": "主题", "value": "充电"}],
    )

    assert matched == 1


def test_重复请求同一对只算一次():
    """模型把同一对写两遍不该把权重翻倍。"""
    metadata = _metadata(主题="充电")
    wanted = [{"field": "主题", "value": "充电"}, {"field": "主题", "value": "充电"}]

    assert tag_match_count(metadata, wanted) == 1


def test_旧块缺tags字段当空():
    """旧记录没有 tags 列时按 []，不报错。"""
    assert tag_match_count({}, [{"field": "品牌", "value": "比亚迪"}]) == 0
    assert source_matches("某手册", empty_filters()) is True


def test_filters_are_set_只看有没有内容():
    """有来源或有标签才算调用方要覆盖意图。"""
    assert filters_are_set(empty_filters()) is False

    with_source = empty_filters()
    with_source["source"] = "手册A"
    assert filters_are_set(with_source) is True

    with_tags = {"source": "", "tags": [{"field": "主题", "value": "充电"}]}
    assert filters_are_set(with_tags) is True


def test_tags_from_filters_把来源也收成一条芯片():
    """来源非空时带 field=source，再追加动态标签。"""
    filters = {"source": "手册A", "tags": [{"field": "品牌", "value": "比亚迪"}]}
    rows = tags_from_filters(filters)

    assert rows == [
        {"field": "source", "value": "手册A"},
        {"field": "品牌", "value": "比亚迪"},
    ]
    assert tags_from_filters(empty_filters()) == []
