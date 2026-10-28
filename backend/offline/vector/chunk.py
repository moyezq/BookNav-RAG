# -*- coding: utf-8 -*-
"""
知识块切分。

使用前注意事项：
- 共享入口是 chunks_from_chunkable_source：有目录区间按标题拼块，否则按段/页。
- 有印刷目录时优先按目录标题拼块，不要先按段切开再丢掉标题。
- 正文页按目录标题分页在本文件，不从抽取方式 import。
- 无目录时退回按空行分段累加；token 按「中文约 2 字 1 token」粗估，不是 tiktoken。
- PDF 无目录时若文本带「=== 第N页 ===」页眉，按页组块（本仓 PDF 抽页直出 pages，不带这种页眉，认不出就退回按段）。
"""

from __future__ import annotations

import re


def estimate_tokens_by_chars(text: str) -> int:
    """按字符粗估 token：中文约 2 字 1 token。"""
    return max(1, len(text) // 2)


def split_doc_text_into_chunks(doc_text: str, max_doc_tokens: int = 1800) -> list[str]:
    """
    按空行分段累积分块，超过 max_doc_tokens 就切开。

    功能：给没有目录结构的纯文本用。
    应用场景：手册抽不出印刷目录时的退路；KV 实验脚本按 kind 分块也调它（还没接进流程）。
    输入：doc_text — 原文；max_doc_tokens — 单块上限（粗估 token）。
    输出：块文本列表；原文为空时 []。
    """
    paragraphs = [part.strip() for part in (doc_text or "").split("\n\n") if part and part.strip()]
    if not paragraphs:
        return []
    chunks = []
    current_text = ""
    for paragraph in paragraphs:
        if not current_text:
            current_text = paragraph
            continue
        next_text = current_text + "\n\n" + paragraph
        if estimate_tokens_by_chars(next_text) > max_doc_tokens:
            chunks.append(current_text)
            current_text = paragraph
        else:
            current_text = next_text
    if current_text:
        chunks.append(current_text)
    return chunks


def split_pdf_text_into_chunks(doc_text: str, pages_per_chunk: int = 5, max_doc_tokens: int = 1800) -> list[str]:
    """
    按「=== 第N页 ===」切页，再按约 pages_per_chunk 页一组。

    功能：一组仍超上限则按段再切。认不出页眉则退回按段分块。
    应用场景：带该页眉的分页 PDF 原文、没有印刷目录可用时；本仓 PDF 抽页直出 pages，此路只服务历史 text 形态。
    输入：doc_text — 带页眉的原文；pages_per_chunk — 每块页数；max_doc_tokens — 上限。
    输出：块文本列表。
    """
    header_pattern = re.compile(r"^=+\s*第\s*(\d+)\s*页\s*=+\s*$")
    lines = (doc_text or "").splitlines()
    pages = []
    current_lines = []
    seen_header = False

    for line in lines:
        if header_pattern.match(line.strip()):
            if current_lines:
                page_text = "\n".join(current_lines).strip()
                if page_text:
                    pages.append(page_text)
            current_lines = [line.strip()]
            seen_header = True
            continue
        if seen_header:
            current_lines.append(line.rstrip())

    if current_lines:
        page_text = "\n".join(current_lines).strip()
        if page_text:
            pages.append(page_text)

    if not pages:
        return split_doc_text_into_chunks(doc_text, max_doc_tokens=max_doc_tokens)

    groups = []
    buffer = []
    for page in pages:
        buffer.append(page)
        if len(buffer) >= pages_per_chunk:
            groups.append("\n\n".join(buffer))
            buffer = []
    if buffer:
        groups.append("\n\n".join(buffer))

    chunks = []
    for group_text in groups:
        if estimate_tokens_by_chars(group_text) > max_doc_tokens:
            chunks.extend(split_doc_text_into_chunks(group_text, max_doc_tokens=max_doc_tokens))
        else:
            chunks.append(group_text)
    return chunks


def split_text_for_kind(doc_text: str, kind: str, max_doc_tokens: int = 1800) -> list[str]:
    """
    按文件类型选退路切分。

    功能：kind==pdf 走按页组块，其余按段。
    应用场景：没有目录区间时。
    输入：doc_text — 原文；kind — 扩展名或 pdf；max_doc_tokens — 上限。
    输出：块文本列表。
    """
    if kind == "pdf":
        return split_pdf_text_into_chunks(doc_text, max_doc_tokens=max_doc_tokens)
    return split_doc_text_into_chunks(doc_text, max_doc_tokens=max_doc_tokens)


def page_to_int(page_info: dict) -> int | None:
    """读页对象上的 page 字段；缺或非数字则 None。"""
    raw = page_info.get("page")
    if isinstance(raw, int):
        return raw
    text = str(raw or "").strip()
    if not text.isdigit():
        return None
    return int(text)


NAVIGATION_TOC = "目录"
NAVIGATION_INDEX = "索引"
NAVIGATION_CHAPTER_COVER = "章扉页"


def is_toc_line(line: str) -> bool:
    """
    这一行是点线目录行：有 `.`/`…` 引导点，且以页码结尾。

    功能：章扉页会复述一遍小节目录，靠「点线 + 尾部页码」把这种行认出来。
    应用场景：navigation_page_reason 数目录行；verify_ingest.py 反过来用它算残留正文行。
    输入：line — 一行文本（已 strip）。
    输出：True 表示是目录行。
    """
    # 14**| 这种加粗页码也算目录行
    return bool(re.search(r"\.{3,}|…{2,}", line) and re.search(r"\d+\**\s*\|?\s*$", line))


def navigation_page_reason(text: str) -> str | None:
    """
    这一页为什么不算正文。

    功能：页首 80 字含「目 录」→ 目录；含「图片索引 / 字母索引」→ 索引；
          点线目录行至少 5 行且过半 → 章扉页；都不像 → None（当正文）。
    应用场景：assign_content_to_catalogue 按目录分页前丢导航页；入库验收口按原因给丢页贴标签。
    输入：text — 该页原始 markdown。
    输出：NAVIGATION_TOC / NAVIGATION_INDEX / NAVIGATION_CHAPTER_COVER；是正文则 None。
    """
    head = (text or "")[:80]
    if re.search(r"目\s*录", head):
        return NAVIGATION_TOC
    if "图片索引" in head or "字母索引" in head:
        return NAVIGATION_INDEX
    content_lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not content_lines:
        return None
    toc_like = sum(1 for line in content_lines if is_toc_line(line))
    if toc_like >= 5 and toc_like >= len(content_lines) * 0.5:
        return NAVIGATION_CHAPTER_COVER
    return None


def is_navigation_page(text: str) -> bool:
    """
    这一页是目录、图片索引、字母索引或章扉页，不当正文。

    功能：navigation_page_reason 的布尔门面（导航页判据只有那一份实现）。
    应用场景：assign_content_to_catalogue 按目录分页前丢掉导航页。
    输入：text — 该页原始 markdown。
    输出：True 表示跳过。
    """
    return navigation_page_reason(text) is not None


def clean_page_text(text: str, strip_images: bool = True) -> str:
    """
    压空白、去标题井号和简单 HTML；汉字之间的抽取空格去掉。可选去掉 markdown 图片。

    功能：**段边界保留**——页内空行仍是空行（`\\n\\n`），段内换行压成一个空格。
          分段累积分块（split_doc_text_into_chunks 按 `\\n\\n` 切）靠的就是这个边界；
          清洗时把换行一律压平，会让「按段累积分块」永不生效（P0·6）。
    应用场景：assign_content_to_catalogue 按目录分页前逐页清洗。
    输入：text — 该页原始 markdown；strip_images — 是否去掉 markdown 图片。
    输出：段间是空行、段内没有换行的页正文；清洗后为空则空串。
    """
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.S)
    text = re.sub(r"</?(?:u|sup|sub|mark)\s*/?>", "", text, flags=re.I)
    # <br> 是段内换行，不能当零宽字符删掉（会把英文相邻两词粘成一个）
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I).replace("\\n", "\n")

    # 空行分段：连续非空行算同一段，空行收束上一段
    paragraphs = []
    buffer = []
    for line in text.split("\n"):
        if line.strip():
            buffer.append(line.strip())
            continue
        if buffer:
            paragraphs.append(buffer)
            buffer = []
    if buffer:
        paragraphs.append(buffer)

    cleaned = []
    for lines in paragraphs:
        paragraph = " ".join(lines)
        paragraph = paragraph.replace("**", "").replace("#", "").replace("-----", "")
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        paragraph = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", paragraph)
        if strip_images:
            paragraph = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", paragraph).strip()
        if paragraph:
            cleaned.append(paragraph)
    return "\n\n".join(cleaned)


def assign_content_to_catalogue(
    content_pages: list[dict],
    catalogue_ranges: list[dict],
) -> list[dict]:
    """
    把正文页按页码放进目录区间。

    功能：一页只归第一条命中的区间。目录、图片索引、字母索引、章扉页不归入。
    应用场景：向量层按目录标题拼知识块之前。
    输入：content_pages — 可分块原文的 pages；catalogue_ranges — 抽取方式给出的目录区间。
    输出：[{page, catalogue_title, text}, ...]。
    """
    rows = []
    for page_info in content_pages:
        page_no = page_to_int(page_info)
        if page_no is None:
            continue
        raw_text = page_info.get("text", "")
        if is_navigation_page(raw_text):
            continue
        text = clean_page_text(raw_text)
        if not text:
            continue
        for catalogue in catalogue_ranges:
            if catalogue["start_page"] <= page_no < catalogue["end_page"]:
                rows.append(
                    {
                        "page": page_no,
                        "catalogue_title": catalogue["title"],
                        "text": text,
                    }
                )
                break
    return rows


def _assemble_pieces_with_pages(texts: list[str], pages: list, max_doc_tokens: int) -> list[tuple]:
    """
    把同一节的逐页正文收成 (正文, 页码) 列表，页码跟着切出来的每一块走。

    功能：先按「页/段边界」把每页拆成不超过上限的小段，再按顺序累加装块；
          每块只记自己真正装进去的那几页，不再把整节页码复制给每个子块。
    应用场景：有印刷目录时的分块。这是页精度（引用页码与评测页判据）的唯一来源。
    输入：texts — 同一目录标题下逐页正文（已清洗）；pages — 与 texts 一一对应的页号；
          max_doc_tokens — 单块上限（粗估 token）。
    输出：[(块正文, [页号, ...]), ...]；都为空时 []。
    """
    if not texts:
        return []

    # 页间用分隔符隔开，这样同一页内部的空行分段不会被误认成翻页
    sentinel = "\x00"
    segments = []
    for page_no, text in zip(pages, texts):
        body = text or ""
        if not body:
            continue
        if estimate_tokens_by_chars(body) <= max_doc_tokens:
            segments.append((body, page_no))
            continue
        # 单页自身超限：按空行再拆，拆出的每段仍算这一页
        for piece in split_doc_text_into_chunks(body, max_doc_tokens=max_doc_tokens):
            segments.append((piece, page_no))
    if not segments:
        return []

    # 相邻且同页的段用空行还原（等于原来逐页 join），翻页处用分隔符
    pieces = []
    for index, (body, page_no) in enumerate(segments):
        text = body
        if index + 1 < len(segments) and segments[index + 1][1] == page_no:
            text += "\n\n"
        elif index + 1 < len(segments):
            text += sentinel
        pieces.append(text)

    packed = []
    current_text = ""
    current_pages = []
    for text, (_, page_no) in zip(pieces, segments):
        candidate = current_text + text
        if current_text and estimate_tokens_by_chars(candidate) > max_doc_tokens:
            packed.append((current_text, current_pages))
            current_text = text
            current_pages = [page_no]
        else:
            current_text = candidate
            if page_no not in current_pages:
                current_pages.append(page_no)
    if current_text:
        packed.append((current_text, current_pages))
    return [(text.replace(sentinel, "\n\n"), list(pages_of_piece)) for text, pages_of_piece in packed]


def chunks_from_assigned_rows(
    assigned_rows: list[dict],
    source_name: str,
    tags: list,
    knowledge_type: str,
    max_doc_tokens: int = 1800,
) -> list[dict]:
    """
    同一目录标题下的正文页拼成知识块；单块超限再按段切开。

    功能：写入约定字段 source/pages，以及过滤标签 tags。
          页码是**块级**的：超限切出的每个子块只带自己覆盖的页，不带整节页。
    应用场景：有印刷目录时入库。
    输入：assigned_rows — 本文件 assign_content_to_catalogue 产出；source_name — 文档身份；tags — 过滤标签。
    输出：[{id, text, metadata}, ...]。
    """
    grouped = []
    current_title = None
    current_pages = []
    current_texts = []
    for row in assigned_rows:
        title = row["catalogue_title"]
        if current_title is None:
            current_title = title
        if title != current_title:
            grouped.append((current_title, current_pages, current_texts))
            current_title = title
            current_pages = []
            current_texts = []
        current_pages.append(row["page"])
        current_texts.append(row["text"])
    if current_title is not None:
        grouped.append((current_title, current_pages, current_texts))

    chunks = []
    serial = 1
    for title, pages, texts in grouped:
        for piece_index, (piece_text, piece_pages) in enumerate(
            _assemble_pieces_with_pages(texts, pages, max_doc_tokens)
        ):
            chunk_id = f"{source_name}-{serial:04d}"
            serial += 1
            chunks.append(
                {
                    "id": chunk_id,
                    "text": piece_text,
                    "metadata": {
                        "source": source_name,
                        "tags": tags,
                        "knowledge_type": knowledge_type,
                        "catalogue_title": title,
                        "pages": piece_pages,
                        "part": piece_index,
                    },
                }
            )
    return chunks


def chunks_from_plain_text(
    doc_text: str,
    kind: str,
    source_name: str,
    tags: list,
    max_doc_tokens: int = 1800,
) -> list[dict]:
    """
    无目录时的知识块：按 kind 切分，加上约定 metadata。

    功能：每块一个 id；source/pages 进约定；tags 当过滤标签。
    应用场景：目录行为空、或以后接任意纯文本。
    输入：doc_text — 原文；kind — pdf 或其它；source_name — 文档身份；tags — 过滤标签。
    输出：[{id, text, metadata}, ...]。
    """
    pieces = split_text_for_kind(doc_text, kind, max_doc_tokens=max_doc_tokens)
    chunks = []
    for index, piece in enumerate(pieces, start=1):
        chunks.append(
            {
                "id": f"{source_name}-{index:04d}",
                "text": piece,
                "metadata": {
                    "source": source_name,
                    "tags": tags,
                    "pages": [],
                    "part": index,
                },
            }
        )
    return chunks


def chunks_from_chunkable_source(source_obj: dict, max_doc_tokens: int = 1800) -> list[dict]:
    """
    可分块原文 → 知识块约定。

    功能：有 pages 且有 catalogue_ranges 则按目录标题拼块；否则按 kind 切纯文本。
    应用场景：向量层读可分块原文之后。
    输入：source_obj — 符合约定的可分块原文（source/kind，pages 或 text）；max_doc_tokens — 单块上限。
    输出：[{id, text, metadata}, ...]。
    """
    source_name = source_obj["source"]
    kind = source_obj["kind"]
    tags = source_obj["tags"] if "tags" in source_obj else []
    knowledge_type = (
        source_obj["knowledge_type"] if "knowledge_type" in source_obj and source_obj["knowledge_type"] else ""
    )
    ranges = source_obj["catalogue_ranges"] if "catalogue_ranges" in source_obj else []
    pages = source_obj["pages"] if "pages" in source_obj else None
    if pages and ranges:
        assigned = assign_content_to_catalogue(pages, ranges)
        return chunks_from_assigned_rows(
            assigned, source_name, tags, knowledge_type, max_doc_tokens=max_doc_tokens
        )
    if pages:
        joined = "\n\n".join(page["text"] or "" for page in pages)
        return chunks_from_plain_text(joined, kind, source_name, tags, max_doc_tokens=max_doc_tokens)
    return chunks_from_plain_text(
        source_obj["text"], kind, source_name, tags, max_doc_tokens=max_doc_tokens
    )
