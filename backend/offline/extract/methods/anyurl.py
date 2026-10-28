# -*- coding: utf-8 -*-
"""
anyurl2text HTTP 抽取客户端（预留实现文件，零调用，还没接进流程）。

使用前注意事项：
- 曾 POST `/rag/extract/file`；该口已随外仓下线，手册抽取改由本仓 methods/pdf.py。
- 路由不再登记本文件；流程不 import。
- 误调 extract_content 会抛 RuntimeError，不发 HTTP。
"""

from __future__ import annotations


def extract_content(probe: dict) -> dict:
    """
    退役实现：不再调外仓。

    功能：拒绝调用并说明解耦原因。
    应用场景：误登记或旧脚本直接 import 时。
    输入：probe — 基础信息（忽略）。
    输出：无。
    异常：恒抛 RuntimeError。
    """
    raise RuntimeError(
        "methods/anyurl.py 已退役：外仓 /rag/extract/file 已下线；"
        "手册 PDF 请走 methods/pdf.py（路由仅登记 pdf）"
    )
