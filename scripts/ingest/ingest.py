"""
入库：原始页 → 可分块原文 → 向量库。不问答。不调 pymupdf4llm。

原理（按系统流程）
  离线完整链路是：PDF → pymupdf4llm 原始页 → 目录/标签/可分块原文 → 向量库。
  本脚本从 raw_pages.json 接着跑：建印刷目录、抽过滤标签、写出可分块原文，再分块写 LanceDB。
  找不到原始页就失败，不回头抽 PDF。抽页走 scripts/ingest/extract_pdf.py。

  例子（名单就是下面的 PROCESS_ONLY；空元组 = 已有 raw_pages.json 的来源全部）
    PROCESS_ONLY=() → 扫全部原始页，每份写出 chunkable_source.json 再写入 LanceDB。
    改分块或标签只需重跑本脚本，不必再抽页。

怎么跑（本文件入口；编号和上面「一条样本」不是同一套）
  职责：按 PROCESS_ONLY 选原始页，逐份建目录、抽标签、写可分块原文、写向量库。
  入口步骤：
    1. list_raw_page_paths — 扫 backend/data/result/<stem>/raw_pages.json。
    2. load_raw_pages — 读 source、name、pages。
    3. build_catalogue — 印刷目录 → 区间；有区间则方式名 handbook。
    4. infer_tags — 正文前若干页 → tags。
    5. write_chunkable — → chunkable_source.json。
    6. index_chunks — 可分块原文 → step4_chunks.json 与 LanceDB chunks 表。
  输入：backend/data/result/<stem>/raw_pages.json。PROCESS_ONLY 非空则只跑名单里的那几份。
  前置：已跑 extract_pdf.py；python -m service.embed.server 已起。
  产出：每来源一份 chunkable_source.json、step4_chunks.json；向量写入 backend/data/lancedb。
  对外：report_empty_tags。
  跑法：仓库根 uv run python scripts/ingest/ingest.py
        整库入库可能超过 5 分钟，RUN_LIMIT_SEC=1800。
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
from backend.conf import RESULT_DIR, SKIP_DIR_NAMES
from backend.offline.extract.catalogue import CatalogueBuild
from backend.offline.extract.collect import ChunkableCollect
from backend.offline.extract.enrich import TagEnrich
from backend.offline.vector.entrance import run_vector

# ========== 按需修改 ==========
# 非空则只跑名单里的源文件（名字带后缀，与 backend/data/source 下的一致）；空元组则扫已有原始页。
PROCESS_ONLY: Final[tuple[str, ...]] = ()
RUN_LIMIT_SEC: Final[int] = 1800
# ==============================

PROCESS_ONLY_STEMS: Final[frozenset[str]] = frozenset(Path(name).stem for name in PROCESS_ONLY)
RAW_PAGES_NAME: Final[str] = "raw_pages.json"


def list_raw_page_paths() -> list[Path]:
    """
    列出各来源已写出的原始页。

    功能：扫 backend/data/result/<source>/raw_pages.json，跳过 qa。
    应用场景：入库第 1 步。
    输入：无，读 RESULT_DIR。
    输出：按路径排序的 JSON 路径列表。
    异常：一个都没有 → FileNotFoundError（提示先跑 extract_pdf.py）。
    """
    paths = []
    if not RESULT_DIR.is_dir():
        raise FileNotFoundError(
            f"未找到产出目录 {RESULT_DIR}，先跑 python scripts/ingest/extract_pdf.py"
        )
    for folder in sorted(RESULT_DIR.iterdir()):
        if not folder.is_dir() or folder.name in SKIP_DIR_NAMES:
            continue
        source_path = folder / RAW_PAGES_NAME
        if source_path.is_file():
            paths.append(source_path)
    if not paths:
        raise FileNotFoundError(
            f"未找到原始页 {RAW_PAGES_NAME}，先跑 python scripts/ingest/extract_pdf.py"
        )
    return paths


def load_raw_pages(raw_path: Path) -> dict:
    """
    读一份原始页 JSON。

    功能：取出 source、name、pages；缺字段或 pages 空则报错。
    应用场景：入库第 2 步。
    输入：raw_path — raw_pages.json。
    输出：{source, name, pages}。
    异常：缺字段或 pages 空 → ValueError。
    """
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    source = payload.get("source") or raw_path.parent.name
    name = payload.get("name") or f"{source}.pdf"
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"原始页缺少 pages：{raw_path}")
    return {"source": source, "name": name, "pages": pages}


def build_catalogue(pages: list[dict]) -> list[dict]:
    """
    从页列表建印刷目录区间。

    功能：调 CatalogueBuild.build。
    应用场景：入库第 3 步。
    输入：pages — 原始页的 [{page, text}]。
    输出：区间列表；没有目录则 []。
    """
    print("[3] 建目录区间", flush=True)
    ranges = CatalogueBuild().build(pages)
    return ranges


def infer_tags(pages: list[dict], source_name: str) -> list[dict]:
    """
    从正文前若干页抽出过滤标签。

    功能：调 TagEnrich.sample_text 再 infer。
    应用场景：入库第 4 步。
    输入：pages — 原始页；source_name — 文档身份。
    输出：[{field, value}, ...]；抽不出则为 []。
    """
    print("[4] 抽过滤标签", flush=True)
    enricher = TagEnrich()
    sample = enricher.sample_text(pages)
    return enricher.infer(sample, source_name)


def write_chunkable(
    raw: dict,
    *,
    catalogue_ranges: list[dict],
    tags: list[dict],
    result_dir: Path,
) -> Path:
    """
    写出可分块原文。

    功能：有目录区间则方式名 handbook，否则 pdf。
    应用场景：入库第 5 步。
    输入：raw — load_raw_pages 产出；catalogue_ranges / tags；result_dir — 本源目录。
    输出：chunkable_source.json 路径。
    """
    extract_method = "handbook" if catalogue_ranges else "pdf"
    print(f"[5] 写可分块原文 method={extract_method}", flush=True)
    return ChunkableCollect().write(
        source=raw["source"],
        name=raw["name"],
        pages=raw["pages"],
        catalogue_ranges=catalogue_ranges,
        tags=tags,
        extract_method=extract_method,
        result_dir=result_dir,
    )


def index_chunks(chunkable_path: Path) -> Path:
    """
    一份可分块原文走完向量层。

    功能：调 vector.entrance.run_vector。
    应用场景：入库第 6 步。须已起 python -m service.embed.server。
    输入：chunkable_path — chunkable_source.json。
    输出：LanceDB 目录路径。
    """
    print(f"[6] 向量层 {chunkable_path.parent.name}", flush=True)
    out_path = run_vector(chunkable_path)
    print(f"[6] 已写入 {out_path}", flush=True)
    return out_path


def report_empty_tags(chunkable_paths: list[Path]) -> None:
    """
    把 tags 抽空的来源单独列出来。

    功能：读每份可分块原文的 tags，空的收成一行汇总（抽空不算入库失败，
          但必须与「有标签」可区分，否则一份没有标签的文档混在「成功 N」里看不出来）。
    应用场景：入库汇总。
    输入：chunkable_paths — 本次真正入库成功的可分块原文路径。
    输出：无（打印）；读不出来的文件跳过，那属于失败列表的事。
    """
    empty: list[str] = []
    for path in chunkable_paths:
        try:
            tags = json.loads(path.read_text(encoding="utf-8")).get("tags") or []
        except (OSError, json.JSONDecodeError):
            continue
        if not tags:
            empty.append(path.parent.name)
    if empty:
        print(f"标签抽空的来源 {len(empty)} 个（检索时不会按标签加权）：{'、'.join(empty)}", flush=True)


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
    chunkable_done: list[Path] = []

    # 1. 列出已抽出的原始页
    raw_paths = list_raw_page_paths()
    if PROCESS_ONLY:
        raw_paths = [path for path in raw_paths if path.parent.name in PROCESS_ONLY_STEMS]
    if not raw_paths:
        raise FileNotFoundError(
            f"没有可入库的原始页 PROCESS_ONLY={PROCESS_ONLY!r}，"
            f"先跑 python scripts/ingest/extract_pdf.py"
        )
    print(f"ingest sources={len(raw_paths)}", flush=True)

    for raw_path in raw_paths:
        try:
            # 2. 读 raw_pages.json
            raw = load_raw_pages(raw_path)
            print(f"[2] 读原始页 {raw['name']} pages={len(raw['pages'])}", flush=True)
            # 3. 印刷目录 → 区间
            ranges = build_catalogue(raw["pages"])
            # 4. 正文前若干页 → tags
            tags = infer_tags(raw["pages"], raw["source"])
            # 5. 写出 chunkable_source.json
            chunkable_path = write_chunkable(
                raw,
                catalogue_ranges=ranges,
                tags=tags,
                result_dir=raw_path.parent,
            )
            # 6. 分块 + 调 8030 + 写入 LanceDB
            index_chunks(chunkable_path)
            succeeded.append(raw["name"])
            chunkable_done.append(chunkable_path)
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            print(f"失败 {raw_path.parent.name}: {exc}", flush=True)
            failed.append((raw_path.parent.name, f"{type(exc).__name__}: {exc}"))
            continue
    _print_batch_result(succeeded, failed)
    report_empty_tags(chunkable_done)
    raise SystemExit(1 if failed else 0)
