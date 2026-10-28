# -*- coding: utf-8 -*-
"""
抽取层入口：一份源 → 可分块原文。

职责：probe → route → 抽取方式 → enrich → collect。不分块、不向量。
      由 scripts/ingest/ingest.py 调用；本文件无 __main__。
输入：backend/data/source/ 下扩展名在允许清单里的文件（本阶段仅 PDF 本仓抽页；不依赖 anyurl2text）。
产出：backend/data/result/<source>/chunkable_source.json。
"""

from __future__ import annotations

from pathlib import Path

from backend.conf import RESULT_DIR, SOURCE_ALLOWED_SUFFIXES, SOURCE_DIR
from backend.offline.extract.collect import collect
from backend.offline.extract.enrich import infer_ingest_tags, sample_content_text
from backend.offline.extract.probe import probe_file
from backend.offline.extract.route import extract_content


def list_source_files() -> list[Path]:
    """
    列出 backend/data/source 下可入库的源文件。

    功能：只认 SOURCE_ALLOWED_SUFFIXES 里的扩展名；跳过的文件打一行日志，不静默丢弃。
    应用场景：抽取层入口。
    输入：无，读 SOURCE_DIR。
    输出：按路径名排序的 Path 列表。
    异常：一个可入库的都没有 → FileNotFoundError（报出被跳过的文件名与支持的后缀）。
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


def run_extract(source_path: Path) -> Path:
    """
    一份源走完抽取层，写出可分块原文。

    功能：基础信息 → 抽正文 → 标签 → collect。
    应用场景：scripts/ingest/ingest.py 逐个文件调用。
    输入：source_path — backend/data/source 下的文件。
    输出：chunkable_source.json 路径。
    """
    probe = probe_file(source_path)
    method, content = extract_content(probe)
    print(
        f"[抽取] name={probe['name']} size={probe['size_bytes']} method={method}"
    )
    tags = infer_ingest_tags(sample_content_text(content), probe["source"])
    result_dir = RESULT_DIR / probe["source"]
    return collect(probe, content, tags, method, result_dir)
