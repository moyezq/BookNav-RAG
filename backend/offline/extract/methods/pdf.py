# -*- coding: utf-8 -*-
"""
本仓 PDF 抽取：pymupdf4llm 抽页，印刷目录建区间。

使用前注意事项：
- 页码对外一律 PDF 阅读器 1 起；pymupdf4llm 的 metadata.page 是 0 起。
- 原始 chunk 含 Rect，不能 json.dumps，只保留 page/text。
- 目录只认正文里的印刷目录页（带「目 录」表头，或紧接目录页且能解析出目录行的续页），
  不读 PDF 书签，也不收图片索引 / 字母索引。
- 目录页数不固定：从第 1 页向后扫到「连续若干页都不是目录页」为止，不写死页数（TODO P1·9）。
- 目录页码用印刷目录上的数字，当作阅读器页码。
- 不调 anyurl2text。不写内嵌图。
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.conf import EXTRACT_CATALOGUE_PREVIEW_PAGES, EXTRACT_CATALOGUE_STOP_PAGES


def _chunk_to_page_row(item: dict, fallback_index: int) -> dict:
    """把 pymupdf4llm 的一页 chunk 收成 {page:int, text:str}。"""
    text = item.get("text", "") or ""
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("page") is not None:
        page_no = int(metadata["page"]) + 1
    elif isinstance(item.get("page"), int):
        page_no = item["page"]
    else:
        page_no = fallback_index + 1
    return {"page": page_no, "text": text}


def _extract_pdf_pages(pdf_path: Path) -> list:
    """
    用 pymupdf4llm 按页抽出 markdown。

    功能：page_chunks=True；不写内嵌图。
    应用场景：本仓 PDF 抽取方式。
    输入：pdf_path — PDF 路径。
    输出：[{page, text}, ...]。
    异常：PDF 不存在或 pymupdf4llm 失败时向上抛。
    """
    import pymupdf4llm

    md_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        write_images=False,
    )
    if not isinstance(md_chunks, list):
        raise TypeError(f"pymupdf4llm 期望 list，实际为 {type(md_chunks).__name__}")
    return [_chunk_to_page_row(item, index) for index, item in enumerate(md_chunks)]


# pymupdf4llm 偶发把多条「标题....页码」收成一行空格分隔；拆开后再交给 _parse_toc_cell。
_MULTI_TOC_ENTRY = re.compile(r".+?(?:\.{2,}|…{2,}|·{2,})\s*\d+")


def _split_joined_toc_cell(cell: str) -> list[str]:
    """一行里若有 ≥2 条点线目录项则拆开，否则原样返回。"""
    parts = [part.strip() for part in _MULTI_TOC_ENTRY.findall(cell) if part.strip()]
    if len(parts) >= 2:
        return parts
    return [cell]


def _toc_cells(page_text: str) -> list:
    """把一页目录 markdown 拆成单元格。`<br>` 先当换行，点线标题被表格拆开时再拼回去。"""
    text = page_text.replace("**", "").replace("`", "")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    cells = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith("|---") or set(raw_line) <= set("|-: "):
            continue
        if raw_line.startswith("|"):
            pieces = [piece.strip() for piece in raw_line.strip("|").split("|")]
        else:
            pieces = [raw_line]
        for piece in pieces:
            if not piece or piece == "---":
                continue
            buf = ""
            for part in re.split(r"<br\s*/?>", piece):
                part = part.strip()
                if not part:
                    continue
                buf = part if not buf else buf + part
                if re.search(r"\d+\s*$", buf):
                    cells.extend(_split_joined_toc_cell(buf))
                    buf = ""
            if buf:
                cells.extend(_split_joined_toc_cell(buf))
    return cells


def _parse_toc_cell(cell: str):
    """
    单元格 → (标题, 页码, 是否 N-N 节标题)。

    节标题形如 `1-1座椅安全带........14`，和同页的叶子条目重复，后面会丢掉。
    """
    cell = re.sub(r"^#+\s*", "", cell).strip()
    kept_star = False
    matched = re.match(r"^(?P<body>.+?)(?:\.{2,}|…{2,}|·{2,})\s*(?P<page>\d+)\s*$", cell)
    if not matched:
        matched = re.match(r"^(?P<body>.+?)\*(?P<page>\d+)\s*$", cell)
        kept_star = matched is not None
    if not matched:
        return None
    is_section = re.match(r"^\d+-\d+", matched.group("body").strip()) is not None
    title = re.sub(r"^\d+-\d+\s*", "", matched.group("body"))
    title = re.sub(r"[\s.·…\-–—]+$", "", title)
    if kept_star:
        title += "*"
    title = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", title)
    title = title.strip()
    if len(title) < 2 or not re.search(r"[\u4e00-\u9fffA-Za-z]", title):
        return None
    if title.lower().startswith("br"):
        return None
    return title, int(matched.group("page")), is_section


def _parse_catalogue_line(line: str):
    """解析一行目录为 (标题, 起始页)；解析不出则 None。"""
    line = (line or "").strip()
    matched = re.match(r"^(?P<title>.+?)\s+(?P<page>\d+)\s*$", line)
    if not matched:
        return None
    title = matched.group("title").strip()
    if len(title) < 2:
        return None
    return title, int(matched.group("page"))


def _toc_item_count(page_text: str) -> int:
    """这一页能解析出几条「标题…页码」目录行。"""
    count = 0
    for cell in _toc_cells(page_text):
        if _parse_toc_cell(cell):
            count += 1
    return count


def _select_catalogue_pages(pages: list) -> list:
    """
    挑出印刷目录页。

    功能：带「目 录」表头的页一定是目录页；紧接上一页目录页、且能解析出目录行的页算续页
          （续页不一定重复表头）。已挑到目录页后，连续 EXTRACT_CATALOGUE_STOP_PAGES 页都不是就停；
          最多看 EXTRACT_CATALOGUE_PREVIEW_PAGES 页。
    应用场景：extract_content 建目录区间之前。
    输入：pages — 全量页 [{page, text}]。
    输出：目录页子列表，保持原页序；没有目录页则 []。
    为什么动态扫：目录页数不固定，写死「只读前 15 页」时超过 15 页的目录会把后半本书
      并进倒数第二个标题（TODO P1·9）。
    """
    selected = []
    misses = 0
    for page in pages[:EXTRACT_CATALOGUE_PREVIEW_PAGES]:
        text = page["text"] or ""
        has_header = re.search(r"目\s*录", text) is not None
        is_continuation = misses == 0 and bool(selected) and _toc_item_count(text) > 0
        if has_header or is_continuation:
            selected.append(page)
            misses = 0
            continue
        misses += 1
        if selected and misses >= EXTRACT_CATALOGUE_STOP_PAGES:
            break
    return selected


def _extract_catalogue_lines(catalogue_pages: list) -> list:
    """
    从已挑出的目录页抽出「标题 + 页码」行，保持目录顺序。

    功能：丢掉 N-N 节标题（同页已有叶子条目时）；同标题同页去重。页的挑选见 _select_catalogue_pages。
    应用场景：按页抽出之后。
    输入：catalogue_pages — 目录页 [{page, text}]。
    输出：`标题 页码` 字符串列表。没有目录行则 []。
    """
    parsed = []
    seen = set()
    for page in catalogue_pages:
        for cell in _toc_cells(page["text"] or ""):
            item = _parse_toc_cell(cell)
            if not item:
                continue
            title, start_page, is_section = item
            parsed.append({"title": title, "start_page": start_page, "is_section": is_section})
    pages_with_leaf = {item["start_page"] for item in parsed if not item["is_section"]}
    lines = []
    for item in parsed:
        if item["is_section"] and item["start_page"] in pages_with_leaf:
            continue
        key = (item["title"], item["start_page"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{item['title']} {item['start_page']}")
    return lines


def _build_catalogue_ranges(catalogue_lines: list) -> list:
    """
    目录行 → [{title, start_page, end_page}]。

    功能：同一起始页的多条叶子并成一条，标题用 ` / ` 连接；end_page 取下一组的 start；末项用 10**9。
    应用场景：抽目录行之后。
    输入：catalogue_lines — _extract_catalogue_lines 的产出。
    输出：按 start_page 排好的区间列表。
    """
    items = []
    for line in catalogue_lines:
        parsed = _parse_catalogue_line(line)
        if parsed:
            title, start_page = parsed
            items.append({"title": title, "start_page": start_page})
    items = sorted(items, key=lambda item: item["start_page"])
    merged = []
    for item in items:
        if merged and merged[-1]["start_page"] == item["start_page"]:
            titles = merged[-1]["title"].split(" / ")
            if item["title"] not in titles:
                merged[-1]["title"] += " / " + item["title"]
            continue
        merged.append({"title": item["title"], "start_page": item["start_page"]})
    for index, item in enumerate(merged):
        item["end_page"] = merged[index + 1]["start_page"] if index + 1 < len(merged) else 10**9
    return merged


def extract_content(probe: dict) -> dict:
    """
    本仓抽出一份 PDF 的文件内容。

    功能：pymupdf4llm 抽页；前若干页能建目录区间则带 catalogue_ranges。
    应用场景：抽取层 PDF 方式。不调外仓。
    输入：probe — 基础信息，含 path。
    输出：{kind, pages, catalogue_ranges} 或 {kind, pages}。
    异常：文件不存在、抽页失败、正文为空 → RuntimeError。
    """
    path = Path(probe["path"])
    if not path.is_file():
        raise RuntimeError(f"本地文件不存在: {path}")
    print(f"[pdf] pymupdf4llm {path.name}")
    pages = _extract_pdf_pages(path)
    if not pages:
        raise RuntimeError("PDF 抽页结果为空")
    joined = "".join(page["text"] or "" for page in pages).strip()
    if not joined:
        raise RuntimeError("PDF 抽出的正文为空")
    catalogue_pages = _select_catalogue_pages(pages)
    lines = _extract_catalogue_lines(catalogue_pages)
    ranges = _build_catalogue_ranges(lines)
    if lines and ranges:
        print(
            f"[pdf] ok kind=pdf pages={len(pages)} 目录页={len(catalogue_pages)} "
            f"目录条目={len(lines)} 区间={len(ranges)} 覆盖到第 {ranges[-1]['start_page']} 页"
        )
        return {"kind": "pdf", "pages": pages, "catalogue_ranges": ranges}
    print(f"[pdf] ok kind=pdf pages={len(pages)} 目录页={len(catalogue_pages)} 区间=0")
    return {"kind": "pdf", "pages": pages}
