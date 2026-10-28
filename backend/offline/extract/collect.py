# -*- coding: utf-8 -*-
"""
写出可分块原文。

职责：把基础信息、文件内容、内容生成信息收成一份 JSON。对外约定见目标架构第 8 节。
"""

from __future__ import annotations

import json
from pathlib import Path


def collect(
    probe: dict,
    content: dict,
    tags: list,
    extract_method: str,
    result_dir: Path,
    knowledge_type: str = "",
) -> Path:
    """
    写出 chunkable_source.json。

    功能：有 pages 则带 pages 与目录区间；否则带 text。溯源字段不进检索约定。
    应用场景：抽取层最后一步。
    输入：probe / content / tags / extract_method；result_dir — 本源产出目录。
    输出：写出的路径。
    """
    payload = {
        "source": probe["source"],
        "kind": content["kind"],
        "tags": tags,
        "extract_method": extract_method,
        "probe": {
            "name": probe["name"],
            "size_bytes": probe["size_bytes"],
            "ext": probe["ext"],
        },
    }
    if "pages" in content:
        payload["pages"] = content["pages"]
        payload["catalogue_ranges"] = (
            content["catalogue_ranges"] if "catalogue_ranges" in content else []
        )
    else:
        payload["text"] = content["text"]
    if knowledge_type:
        payload["knowledge_type"] = knowledge_type
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / "chunkable_source.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"可分块原文 已写入 {out_path}")
    return out_path
