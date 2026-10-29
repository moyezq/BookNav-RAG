"""
写出可分块原文。

职责：把原始页、目录区间、过滤标签收成一份 JSON。对外约定见目标架构。
对外：ChunkableCollect。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


class ChunkableCollect:
    """原始页 + 目录 + tags → chunkable_source.json。"""

    def write(
        self,
        *,
        source: str,
        name: str,
        pages: Sequence[dict],
        catalogue_ranges: Sequence[dict],
        tags: Sequence[dict],
        extract_method: str,
        result_dir: Path,
    ) -> Path:
        """
        写出 chunkable_source.json。

        功能：有 pages 则带 pages 与目录区间。溯源字段不进检索约定。
        应用场景：ingest.py 在向量层之前。
        输入：source — 文档身份；name — 源文件名；pages — 原始页；
              catalogue_ranges — 可为 []；tags — 可为 []；extract_method — handbook 或 pdf；
              result_dir — 本源产出目录。
        输出：写出的路径。
        """
        payload = {
            "source": source,
            "kind": "pdf",
            "tags": list(tags),
            "extract_method": extract_method,
            "probe": {"name": name},
            "pages": list(pages),
            "catalogue_ranges": list(catalogue_ranges),
        }
        result_dir.mkdir(parents=True, exist_ok=True)
        out_path = result_dir / "chunkable_source.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"可分块原文 已写入 {out_path}", flush=True)
        return out_path
