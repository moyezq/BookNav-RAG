# -*- coding: utf-8 -*-
"""
重排序层：把召回候选截到 TOP_K。不换序。

文件名仍是 overlap：词重叠重排已拆（ADR-0030），这里只剩截断，不要再按名字接第二套打分。
"""

from backend.conf import TOP_K


def rerank_hits(hits: list[dict]) -> list[dict]:
    """
    重排窗口：保持召回顺序，截 TOP_K。

    功能：不换序。截完交给同节邻块、拒答与生成。
          原形参 `question` 属于已拆掉的词重叠重排，没有调用方读它，已去掉。
    应用场景：在线图 rerank 节点；评测口对齐生成窗口。
    输入：hits — 召回候选。
    输出：同形状命中，长度 ≤ TOP_K。
    """
    cut = hits[:TOP_K]
    print(f"rerank in={len(hits)} out={len(cut)}")
    return cut
