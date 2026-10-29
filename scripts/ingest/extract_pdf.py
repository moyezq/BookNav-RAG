"""
抽页：backend/data/source 的 PDF → 原始页 JSON。不建目录、不抽标签、不写向量库。

原理（按系统流程）
  离线完整链路是：PDF → pymupdf4llm 原始页 → 目录/标签/可分块原文 → 向量库。
  本脚本每份源只走到原始页结束。不跑 ingest.py 里的目录、标签、分块。
  pymupdf4llm 是整条链路里最慢的一步，单独落盘后改分块或标签不必重抽页。

  例子（名单就是下面的 PROCESS_ONLY；空元组 = backend/data/source 全部 PDF）
    PROCESS_ONLY=() → 扫全部允许后缀的源，每份写出 raw_pages.json。
    已有 raw_pages.json 的来源会被覆盖。

怎么跑（本文件入口；编号和上面「一条样本」不是同一套）
  职责：按 PROCESS_ONLY 选 PDF，逐份调 PdfPageExtract，写出原始页后退出。
  入口步骤：
    1. list_source_files — 扫 backend/data/source，只留允许后缀。
    2. extract_pages — 一份 PDF → [{page, text}]（页码从 1 起，无 Rect）。
    3. write_raw_pages — → backend/data/result/<stem>/raw_pages.json。
  输入：backend/data/source/ 下的文件。PROCESS_ONLY 非空则只跑名单里的那几份。
  前置：无向量服务、无 Chat。仅 PDF。
  产出：每来源一份 raw_pages.json（source、name、pages）。
  对外：extract_pages、write_raw_pages。
  跑法：仓库根 uv run python scripts/ingest/extract_pdf.py
        整库抽页可能超过 5 分钟，RUN_LIMIT_SEC=1800。
"""

from __future__ import annotations

import faulthandler
import json
import sys
import traceback
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT 在 conf.base；读 yaml 与 .env
from backend.conf import RESULT_DIR
from backend.offline.extract.pdf_page import PdfPageExtract, list_source_files

# ========== 按需修改 ==========
# 非空则只跑名单里的源文件（名字带后缀，与 backend/data/source 下的一致）；空元组则扫全部。
PROCESS_ONLY: Final[tuple[str, ...]] = ()
RUN_LIMIT_SEC: Final[int] = 1800
# ==============================


def extract_pages(source_path: Path) -> list[dict]:
    """
    一份 PDF 抽成页列表。

    功能：调 PdfPageExtract.extract。
    应用场景：抽页第 2 步。
    输入：source_path — backend/data/source 下的 PDF。
    输出：[{page, text}, ...]。
    """
    print(f"[2] 抽页 {source_path.name}", flush=True)
    pages = PdfPageExtract().extract(source_path)
    print(f"[2] {source_path.name} pages={len(pages)}", flush=True)
    return pages


def write_raw_pages(source_path: Path, pages: list[dict]) -> Path:
    """
    把页列表写成原始页 JSON。

    功能：去掉几何信息，只留 source、name、pages。
    应用场景：抽页第 3 步。
    输入：source_path — 源 PDF；pages — extract_pages 的产出。
    输出：raw_pages.json 路径。
    """
    result_dir = RESULT_DIR / source_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / "raw_pages.json"
    payload = {
        "source": source_path.stem,
        "name": source_path.name,
        "pages": pages,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3] 已写入 {out_path}", flush=True)
    return out_path


def _print_batch_result(succeeded: list[str], failed: list[tuple[str, str]]) -> None:
    """打印成功数/失败数，并列出失败项与原因。"""
    print(f"成功 {len(succeeded)} / 失败 {len(failed)}", flush=True)
    for name, reason in failed:
        print(f"  失败项 {name}: {reason}", flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    # 1. 列出可抽页的 PDF
    source_paths = list_source_files()
    if PROCESS_ONLY:
        source_paths = [path for path in source_paths if path.name in PROCESS_ONLY]
    if not source_paths:
        raise FileNotFoundError(f"没有可抽页的源文件 PROCESS_ONLY={PROCESS_ONLY!r}")
    print(f"extract_pdf sources={len(source_paths)}", flush=True)

    for source_path in source_paths:
        try:
            # 2. pymupdf4llm 抽页（页码从 1 起，无 Rect）
            pages = extract_pages(source_path)
            # 3. 写出 backend/data/result/<stem>/raw_pages.json
            write_raw_pages(source_path, pages)
            succeeded.append(source_path.name)
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            print(f"失败 {source_path.name}: {exc}", flush=True)
            failed.append((source_path.name, f"{type(exc).__name__}: {exc}"))
            continue
    _print_batch_result(succeeded, failed)
    raise SystemExit(1 if failed else 0)
