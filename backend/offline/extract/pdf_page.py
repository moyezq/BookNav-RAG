"""
PDF 抽页：pymupdf4llm 按页抽出 markdown，转成阅读器页码。

使用前注意事项：
- 页码对外一律 PDF 阅读器 1 起；pymupdf4llm 的 metadata.page 是 0 起。
- 原始 chunk 含 Rect，不能 json.dumps，只保留 page/text。
- 不建印刷目录、不抽标签、不写可分块原文。不写内嵌图。
- 不调 anyurl2text。
对外：PdfPageExtract、list_source_files。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from backend.conf import SOURCE_ALLOWED_SUFFIXES, SOURCE_DIR


def list_source_files() -> list[Path]:
    """
    列出 backend/data/source 下可抽页的源文件。

    功能：只认 SOURCE_ALLOWED_SUFFIXES 里的扩展名；跳过的文件打一行日志，不静默丢弃。
    应用场景：extract_pdf.py 第 1 步。
    输入：无，读 SOURCE_DIR。
    输出：按路径名排序的 Path 列表。
    异常：一个可抽页的都没有 → FileNotFoundError（报出被跳过的文件名与支持的后缀）。
    """
    all_files = sorted(path for path in SOURCE_DIR.iterdir() if path.is_file())
    paths = [path for path in all_files if path.suffix.lower() in SOURCE_ALLOWED_SUFFIXES]
    skipped = [path.name for path in all_files if path.suffix.lower() not in SOURCE_ALLOWED_SUFFIXES]
    if skipped:
        print(f"[抽取] 跳过格式不支持的源 {len(skipped)} 个：{', '.join(skipped)}")
    if not paths:
        supported = " ".join(sorted(SOURCE_ALLOWED_SUFFIXES))
        raise FileNotFoundError(
            f"未找到可入库的源文件（只支持 {supported}），请放到 {SOURCE_DIR}"
        )
    return paths


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


class PdfPageExtract:
    """一份 PDF → [{page, text}, ...]。不写文件。"""

    def extract(self, pdf_path: Path) -> list[dict]:
        """
        用 pymupdf4llm 按页抽出 markdown。

        功能：page_chunks=True；不写内嵌图；页码转成阅读器 1 起。
        应用场景：extract_pdf.py 抽页步骤。
        输入：pdf_path — PDF 路径。
        输出：[{page, text}, ...]。
        异常：文件不存在、抽页失败、正文为空 → RuntimeError。
        """
        if not pdf_path.is_file():
            raise RuntimeError(f"本地文件不存在: {pdf_path}")
        print(f"[pdf] pymupdf4llm {pdf_path.name}", flush=True)
        pages = self._extract_pdf_pages(pdf_path)
        if not pages:
            raise RuntimeError("PDF 抽页结果为空")
        joined = "".join(page["text"] or "" for page in pages).strip()
        if not joined:
            raise RuntimeError("PDF 抽出的正文为空")
        print(f"[pdf] ok pages={len(pages)}", flush=True)
        return pages

    def _extract_pdf_pages(self, pdf_path: Path) -> list[dict]:
        """调用 pymupdf4llm.to_markdown，把每页收成 {page, text}。"""
        import pymupdf4llm

        md_chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            write_images=False,
        )
        if not isinstance(md_chunks, Sequence) or isinstance(md_chunks, (str, bytes)):
            raise TypeError(f"pymupdf4llm 期望 list，实际为 {type(md_chunks).__name__}")
        return [_chunk_to_page_row(item, index) for index, item in enumerate(md_chunks)]
