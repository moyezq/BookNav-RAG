# -*- coding: utf-8 -*-
"""
抽取路由：本阶段只登记本仓 PDF 抽页。

使用前注意事项：有印刷目录则方式名 handbook。本仓 PDF 无目录记 pdf。
      非 PDF 不走外仓；anyurl 是预留实现，还没接进流程。
      增方式：加 methods 文件并在 METHOD_EXTRACTORS 写一行。
      不按样本文件名切换。
"""

from __future__ import annotations

METHOD_HANDBOOK = "handbook"
METHOD_PDF = "pdf"
METHOD_ANYURL = "anyurl"


def extract_content(probe: dict) -> tuple[str, dict]:
    """
    抽出一份文件的内容，并给出方式名。

    功能：ext 为 pdf 走本仓抽页；有目录区间则记 handbook。非 PDF 拒绝。
    应用场景：probe 之后、enrich 之前。
    输入：probe — probe_file 产出。
    输出：(方式名, 文件内容)。
    异常：非 pdf → RuntimeError。
    """
    ext = (probe.get("ext") or "").lower()
    if ext != "pdf":
        raise RuntimeError(
            f"本阶段抽取只支持 PDF 手册，不支持 .{ext or '?'}；"
            "anyurl2text 抽取口已解耦，methods/anyurl.py 是还没接进流程的预留实现"
        )
    content = extract_content_for(METHOD_PDF, probe)
    if "catalogue_ranges" in content and content["catalogue_ranges"]:
        return METHOD_HANDBOOK, content
    return METHOD_PDF, content


def extract_content_for(method: str, probe: dict) -> dict:
    """
    按登记表调用对应方式的 extract_content。

    功能：只认 METHOD_EXTRACTORS 里的名字。
    应用场景：抽取层入口。
    输入：method — 已登记的方式名；probe — 基础信息。
    输出：文件内容（pages 或 text）。
    异常：未登记 → ValueError。
    """
    if method not in METHOD_EXTRACTORS:
        raise ValueError(f"未登记的抽取方式: {method}")
    return METHOD_EXTRACTORS[method](probe)


def _pdf_extract(probe: dict) -> dict:
    """登记用：调 pdf.extract_content。"""
    from backend.offline.extract.methods.pdf import extract_content as run

    return run(probe)


METHOD_EXTRACTORS = {
    METHOD_PDF: _pdf_extract,
}
