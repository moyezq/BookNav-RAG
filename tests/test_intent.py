# -*- coding: utf-8 -*-
"""
意图层解析与收窄的单元测试（纯函数，不调 LLM）。

职责：锁定 backend/online/intent/classify.py 里三处不该被模型带偏的地方——
      1. 模型输出脏 JSON 时的兜底（分类拿不准 → retrieve；筛选认不出 → 不过滤）；
      2. 筛选只能取库内已出现的来源与标签对（_constrain_to_catalog）；
      3. 问句里直接点名来源/标签值时的词面匹配规则（含它现在只能整段匹配的局限）。

应用场景：改 classify.py 或提示词之后跑 python -m pytest tests/test_intent.py。
"""

from __future__ import annotations

from backend.online.intent.classify import (
    _constrain_to_catalog,
    _lexical_filters,
    _longest_catalog_hit,
    _normalize_question,
    _parse_filter_json,
    _parse_route_json,
    classify_route,
    is_identity_question,
)

CATALOG = {
    "sources": ["汉EV闪充版用户手册260827", "宋Ultra EV用户手册260915"],
    "tags": [
        {"field": "品牌", "value": "比亚迪"},
        {"field": "车型", "value": "汉EV"},
        {"field": "主题", "value": "充电"},
    ],
}

# ========== 分类：解析与兜底 ==========


def test_分类JSON正常解析():
    assert _parse_route_json('{"route":"chitchat"}') == "chitchat"
    assert _parse_route_json('{"route":"retrieve"}') == "retrieve"


def test_分类输出认不出时按检索():
    """拿不准输出 retrieve，宁可多检索。"""
    assert _parse_route_json("我不知道") == "retrieve"
    assert _parse_route_json('{"route":"闲聊"}') == "retrieve"
    assert _parse_route_json('```json\n{"route":"chitchat"}\n```') == "chitchat"


def test_套话整句判闲聊且不调模型():
    """整句套话走词表，不该打 LLM。"""
    assert classify_route("你好") == "chitchat"
    assert classify_route("谢谢！") == "chitchat"


def test_身份追问识别():
    """问资料来源/品牌/车型算身份追问，不是问助手本人。"""
    assert is_identity_question("你的资料来源于哪个品牌") is True
    assert is_identity_question("你是谁") is False


def test_问句归一化去尾部标点():
    assert _normalize_question("  你好！！ ") == "你好"


# ========== 筛选：解析、收窄、词面匹配 ==========


def test_筛选JSON认不出则不过滤():
    assert _parse_filter_json("不是 JSON") == {"source": "", "tags": []}


def test_筛选解析丢掉source字段的同名标签():
    """标签里出现 field=source 要丢掉：来源只走 source 字段。"""
    raw = '{"source":"手册A","tags":[{"field":"source","value":"手册A"},{"field":"品牌","value":"比亚迪"}]}'
    parsed = _parse_filter_json(raw)

    assert parsed["source"] == "手册A"
    assert parsed["tags"] == [{"field": "品牌", "value": "比亚迪"}]


def test_收窄只保留库内已有的来源与标签对():
    """模型编出来的值一律丢掉。"""
    raw = {
        "source": "库里没有的手册",
        "tags": [
            {"field": "品牌", "value": "比亚迪"},
            {"field": "品牌", "value": "保时捷"},
            {"field": "主题", "value": "充电"},
        ],
    }
    constrained = _constrain_to_catalog(raw, CATALOG)

    assert constrained["source"] == ""
    assert constrained["tags"] == [
        {"field": "品牌", "value": "比亚迪"},
        {"field": "主题", "value": "充电"},
    ]


def test_词面匹配取最长的那个来源名():
    """更长的来源名优先，避免短串误伤。"""
    question = "宋Ultra EV用户手册260915里后视镜怎么折叠"
    assert _longest_catalog_hit(question, CATALOG["sources"]) == "宋Ultra EV用户手册260915"


def test_词面匹配要求问句逐字包含来源名():
    """已知局限（P2·23）：用户说「汉EV手册」匹配不到，只能说全名。"""
    assert _longest_catalog_hit("汉EV手册里后视镜怎么折叠", CATALOG["sources"]) == ""


def test_词面筛选同字段多值都收():
    """ADR-0035：同字段的多个值都收下来加权，不再去重成一条（去重是为强制筛选服务的）。"""
    catalog = {
        "sources": CATALOG["sources"],
        "tags": [
            {"field": "主题", "value": "充电"},
            {"field": "主题", "value": "安全"},
        ],
    }
    filters = _lexical_filters("充电和安全都说说", catalog)

    assert {tag["value"] for tag in filters["tags"]} == {"安全", "充电"}


def test_词面筛选忽略标签值里的空格():
    """实测注意点：库里抽出的值是「问界 新M5 增程版」——问句里空格写法不同不该让加权落空。"""
    catalog = {"sources": [], "tags": [{"field": "车型", "value": "问界 新M5 增程版"}]}

    filters = _lexical_filters("问界新M5增程版的智驾怎么开", catalog)

    assert filters["tags"] == [{"field": "车型", "value": "问界 新M5 增程版"}]


def test_词面筛选只说了标签值的一部分就不命中():
    """已知局限（TODO P1·34）：用户只说「问界新M5」，库里的值是「问界 新M5 增程版」→ 落空。

    这一路只做**加权**、不强制筛选，落空只少一点加分；要更宽就得改成前缀/子串匹配，
    那是 ADR-0035 的判据变更，先记在 P1·34，不在本轮私自改。
    """
    catalog = {"sources": [], "tags": [{"field": "车型", "value": "问界 新M5 增程版"}]}

    assert _lexical_filters("问界新M5的智驾怎么开", catalog)["tags"] == []


def test_词面筛选忽略过短的标签值():
    """短于 TAG_LEXICAL_MIN_CHARS 的值（单个字）不参与匹配，避免误伤。"""
    catalog = {"sources": [], "tags": [{"field": "主题", "value": "车"}]}

    assert _lexical_filters("这车怎么充电", catalog)["tags"] == []


def test_问句没点名时不给筛选():
    """没点名来源也没有标签值命中 → 全空，交给模型或直接不过滤。"""
    filters = _lexical_filters("后视镜怎么折叠", CATALOG)

    assert filters == {"source": "", "tags": []}
