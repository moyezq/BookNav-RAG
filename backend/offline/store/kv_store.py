# -*- coding: utf-8 -*-
"""
KV 条目的本地 JSON 存储。提问口不读；实验脚本可写。

状态：预留实现文件（先放着，本阶段不接进流程）。线上主路径与 tests 都不 import 它（零调用），唯一写入方是
      `scripts/url_to_knowledge.py`。与 `backend/offline/extract/methods/vlm_ocr.py` 同属
      「已登记、不接进流程」的预留实现：不要删，也不要把 KV 接进召回路径（ADR-0005、ADR-0030）。

使用前注意事项：
- 字段约定：id、key、value、metadata.source；metadata.tags 可选。
- 不要把条目再编进向量库。精确匹配走本文件。
"""

from __future__ import annotations

import json
from pathlib import Path


class LocalKvStore:
    """一份 KV 条目 JSON。"""

    def __init__(self):
        self.records = []

    def upsert(self, entries: list[dict]) -> None:
        """
        按 id 覆盖写入。

        功能：同 id 后写覆盖先写。
        应用场景：scripts/url_to_knowledge.py 的 write_kv（KV 实验，还没接进流程）。
        输入：entries — [{id, key, value, metadata}, ...]。
        输出：无，写入 self.records。
        """
        by_id = {record["id"]: record for record in self.records}
        for entry in entries:
            by_id[entry["id"]] = {
                "id": entry["id"],
                "key": entry["key"],
                "value": entry["value"],
                "metadata": entry["metadata"],
            }
        self.records = list(by_id.values())

    def save(self, path: Path) -> None:
        """
        写成 JSON 文件。

        功能：写出 records。
        应用场景：KV 抽取结束。
        输入：path — backend/data/result/kv/kv_store.json。
        输出：无。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": self.records}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalKvStore":
        """
        从 JSON 读回 KV 存储。

        功能：认 records。
        应用场景：KV 实验脚本。
        输入：path — kv_store.json。
        输出：已填 records 的 Store。
        异常：文件不存在 → FileNotFoundError。
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        store.records = payload["records"]
        return store
