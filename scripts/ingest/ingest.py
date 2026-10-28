"""
入库：源文件 → 可分块原文 → 向量库。不问答。

原理（按系统流程）
  离线完整链路是：data/source 一份源 → 抽取层可分块原文 → 向量层知识块 → data/lancedb。
  本脚本逐份串行：先抽取再向量。VECTOR_ONLY=True 时跳过抽取，复用已有 chunkable_source.json。
  抽取实现在 offline/extract/，向量实现在 offline/vector/。本文件只编排。

  例子（名单就是下面的 PROCESS_ONLY；空元组 = data/source 全部 PDF）
    PROCESS_ONLY=() → 扫全部允许后缀的源，每份写出 chunkable_source.json 再写入 LanceDB。
    VECTOR_ONLY=True → 不重跑抽取与 enrich，只重跑分块与写库。

怎么跑（本文件入口；编号和上面「一条样本」不是同一套）
  职责：按 PROCESS_ONLY 选源，逐份跑抽取与向量，失败记入列表后继续，最后打印成功/失败数。
  入口步骤：
    1. extract_source — 一份源 → data/result/<stem>/chunkable_source.json（VECTOR_ONLY 时跳过）。
    2. index_chunks — 可分块原文 → step4_chunks.json 与 data/lancedb chunks 表。
  输入：data/source/ 下的文件。PROCESS_ONLY 非空则只跑名单里的那几份。
  前置：python -m embed.server 已起（EMBEDDER_KIND=http）。仅 PDF 本仓抽页。
  产出：每来源一份 chunkable_source.json、step4_chunks.json；向量写入 data/lancedb。
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
import conf  # ROOT 在 conf.base；读 yaml 与 .env

from offline.extract.entrance import list_source_files, run_extract
from offline.vector.entrance import list_chunkable_paths, run_vector

# ========== 按需修改 ==========
# 非空则只跑名单里的源文件（名字带后缀，与 data/source 下的一致）；空元组则扫 data/source 全部。
PROCESS_ONLY: Final[tuple[str, ...]] = ()
# True：跳过抽取，复用已有 chunkable_source.json 只重跑向量层
VECTOR_ONLY: Final[bool] = False
RUN_LIMIT_SEC: Final[int] = 1800
# ==============================

PROCESS_ONLY_STEMS: Final[frozenset[str]] = frozenset(Path(name).stem for name in PROCESS_ONLY)


def extract_source(source_path: Path) -> Path:
    """
    一份源走完抽取层。

    功能：调 extract.entrance.run_extract。
    应用场景：入库第 1 步。
    输入：source_path — data/source 下的文件。
    输出：chunkable_source.json 路径。
    """
    print(f"[1] 抽取 {source_path.name}", flush=True)
    out_path = run_extract(source_path)
    print(f"[1] 已写入 {out_path}", flush=True)
    return out_path


def index_chunks(chunkable_path: Path) -> Path:
    """
    一份可分块原文走完向量层。

    功能：调 vector.entrance.run_vector。
    应用场景：入库第 2 步。须已起 python -m embed.server。
    输入：chunkable_path — chunkable_source.json。
    输出：LanceDB 目录路径。
    """
    print(f"[2] 向量层 {chunkable_path.parent.name}", flush=True)
    out_path = run_vector(chunkable_path)
    print(f"[2] 已写入 {out_path}", flush=True)
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

    if VECTOR_ONLY:
        # 只跑第 2 步：分块 + 调 8030 + 写入 LanceDB
        chunkable_paths = list_chunkable_paths()
        if PROCESS_ONLY:
            chunkable_paths = [
                path for path in chunkable_paths if path.parent.name in PROCESS_ONLY_STEMS
            ]
        if not chunkable_paths:
            raise FileNotFoundError(f"没有可重跑的可分块原文 PROCESS_ONLY={PROCESS_ONLY!r}")
        print(f"ingest VECTOR_ONLY sources={len(chunkable_paths)}", flush=True)
        for chunkable_path in chunkable_paths:
            try:
                index_chunks(chunkable_path)
                succeeded.append(chunkable_path.parent.name)
                chunkable_done.append(chunkable_path)
            except Exception as exc:
                print(traceback.format_exc(), flush=True)
                print(f"失败 {chunkable_path.parent.name}: {exc}", flush=True)
                failed.append((chunkable_path.parent.name, f"{type(exc).__name__}: {exc}"))
                continue
        _print_batch_result(succeeded, failed)
        report_empty_tags(chunkable_done)
        raise SystemExit(1 if failed else 0)

    source_paths = list_source_files()
    if PROCESS_ONLY:
        source_paths = [path for path in source_paths if path.name in PROCESS_ONLY]
    if not source_paths:
        raise FileNotFoundError(f"没有可入库的源文件 PROCESS_ONLY={PROCESS_ONLY!r}")
    print(f"ingest sources={len(source_paths)}", flush=True)
    for source_path in source_paths:
        try:
            # 1. 抽取 → data/result/<stem>/chunkable_source.json
            chunkable_path = extract_source(source_path)
            # 2. 分块 + 调 8030 + 写入 LanceDB
            index_chunks(chunkable_path)
            succeeded.append(source_path.name)
            chunkable_done.append(chunkable_path)
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            print(f"失败 {source_path.name}: {exc}", flush=True)
            failed.append((source_path.name, f"{type(exc).__name__}: {exc}"))
            continue
    _print_batch_result(succeeded, failed)
    report_empty_tags(chunkable_done)
    raise SystemExit(1 if failed else 0)
