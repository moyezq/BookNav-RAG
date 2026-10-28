# -*- coding: utf-8 -*-
"""
向量库 LanceDB 后端与进程内 BM25。

使用前注意事项：
- HashEmbedder 不是语义模型，只为不依赖外部 embedding 服务时把流程跑通。
- HttpEmbedder 认 OpenAI 兼容的 data[].embedding；URL 与模型名由调用方常量传入。
- HttpEmbedder 按 request_batch 切批（0 = 整批一个请求），一批以上时每批打一行 [向量层] 进度：
  整库入库在 CPU 上要十几分钟，没有进度输出会被当成卡死。
- LanceVectorStore 是嵌入式目录后端，不是 Milvus。catalog 在 data/lancedb/，一张 chunks 表。
- 按来源替换写入：先写入 source=__writing__ 暂存行，再删旧来源、写入正式行、删暂存。
  任一步失败时旧数据仍在或暂存可辨（不再「先删后加 → 库空了」）。
- tags 列是 JSON 字符串。catalogue_title 是目录标题，无则空串。旧表缺这两列，须删 catalog 后重入库。
- BM25：拉丁字母与数字按整词切；汉字用相邻二字。与向量检索并列，融合由召回层做。
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import lancedb
import requests

CHUNKS_TABLE = "chunks"
PAYLOAD_COLUMNS = ["id", "text", "source", "pages", "embedder", "tags", "catalogue_title"]
# 入库半截标记：正式来源名不会含这段；问答侧 load 时跳过带此后缀的行
WRITING_MARKER = "::__writing__"

_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]+")


def _staging_source(source: str) -> str:
    """半截写入用的临时来源名。"""
    return f"{source}{WRITING_MARKER}"


def _is_writing_source(source: str) -> bool:
    """是不是入库暂存来源。"""
    return source.endswith(WRITING_MARKER)


class HashEmbedder:
    """字符散列临时占位向量。"""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = "hash"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        把一批文本编成固定维向量。

        功能：按字符码累加到 dim 个桶，再 L2 归一。
        应用场景：EMBEDDER_KIND=hash。
        输入：texts — 知识块正文。
        输出：与 texts 等长的向量列表。
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """单条文本 → 归一化散列向量。"""
        vector = [0.0] * self.dim
        if not text:
            vector[0] = 1.0
            return vector
        for index, char in enumerate(text):
            bucket = (ord(char) * 131 + index) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class HttpEmbedder:
    """OpenAI 兼容 /v1/embeddings。"""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        api_key: str = "",
        timeout_sec: int = 120,
        request_batch: int = 16,
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        # 0 或负数 = 不切批，整批一个请求（只有明确知道总量时才这么用）
        self.request_batch = request_batch if request_batch and request_batch > 0 else 0
        self.name = model_name
        self.dim = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        调 embedding HTTP 接口。

        功能：把 texts 按 request_batch 切成多次 POST，再按顺序拼起来。
              **不把整个库塞进一个请求**：bge-m3 在 CPU 上实测 4.6～6.0 s/段，
              138 段要 631～833s，超过客户端读超时后整批作废，而服务端仍会把那批算完
              ——白等一轮还占着编码锁。批大小取服务端 BATCH_SIZE 的整数倍，
              服务端内部分批与整批发一致，算出来的向量不变。
        应用场景：EMBEDDER_KIND=http；入库整库与在线单条查询都走这里。
        输入：texts — 知识块或查询正文。
        输出：向量列表，顺序与 texts 一致。
        异常：HTTP 非 200 → 向上抛。
        """
        if not texts:
            return []
        size = self.request_batch or len(texts)
        batch_total = math.ceil(len(texts) / size)
        started = time.monotonic()
        vectors: list[list[float]] = []
        for index, start in enumerate(range(0, len(texts), size), start=1):
            vectors.extend(self._post_batch(texts[start : start + size]))
            # 只有多批才报进度：在线单条查询是一批，别给问答进程的日志加噪音
            if batch_total > 1:
                print(
                    f"[向量层] 批 {index}/{batch_total} 已收 {len(vectors)}/{len(texts)} 段 "
                    f"用时 {time.monotonic() - started:.1f}s",
                    flush=True,
                )
        if vectors:
            self.dim = len(vectors[0])
        return vectors

    def _post_batch(self, texts: list[str]) -> list[list[float]]:
        """一次 POST，按 data[].embedding 原样收下（顺序即 texts 顺序）。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.api_url,
            headers=headers,
            json={"model": self.model_name, "input": texts},
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        return [row["embedding"] for row in payload["data"]]


def _sql_literal(value: str) -> str:
    """SQL 字符串字面量，单引号加倍。"""
    return "'" + value.replace("'", "''") + "'"


def _vector_dim(table) -> int:
    """从表 schema 读向量维数，不读 vector 列的值。"""
    return table.schema.field("vector").type.list_size


def _tags_json(tags: list) -> str:
    """过滤标签列表收成表列字符串。"""
    return json.dumps(tags, ensure_ascii=False)


def _tags_from_cell(value) -> list:
    """表列 tags 收成 [{field, value}, ...]。"""
    if value is None or value == "":
        return []
    return json.loads(value)


def _pages_list(value) -> list:
    """把库里的 pages 收成 Python list。"""
    if value is None:
        return []
    return [int(page) for page in value]


def _check_vectors(vectors: list[list[float]], dim: int) -> None:
    """建表前的一致性校验：向量非空、等长、且等于调用方声明的 dim。

    异常：空批、长度不一致、与声明维数不符 → ValueError。
    """
    if not vectors:
        raise ValueError("向量为空，不能建表")
    lengths = {len(vector) for vector in vectors}
    if len(lengths) > 1:
        raise ValueError(f"向量长度不一致：{sorted(lengths)}")
    actual = lengths.pop()
    if actual != dim:
        raise ValueError(
            f"声明维数 {dim} 与实际向量维数 {actual} 不一致：调用方传错了 dim，或换了 embedder"
        )


class LanceVectorStore:
    """data/lancedb 目录里一张 chunks 表。"""

    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_dir))

    def _open_chunks(self):
        """打开 chunks 表。没有表或仍是旧 brand/car 列则失败。"""
        names = self.db.table_names()
        if CHUNKS_TABLE not in names:
            raise FileNotFoundError(f"未找到向量表 {CHUNKS_TABLE}，请先入库：{self.db_dir}")
        table = self.db.open_table(CHUNKS_TABLE)
        column_names = [field.name for field in table.schema]
        if "tags" not in column_names or "catalogue_title" not in column_names:
            raise ValueError(
                f"向量表缺 tags 或 catalogue_title 列。请删掉 {self.db_dir} 后重新入库"
            )
        return table

    def _rows_from_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
        embedder_name: str,
    ) -> list[dict]:
        """知识块与向量收成表行。"""
        rows = []
        for chunk, vector in zip(chunks, vectors):
            metadata = chunk["metadata"]
            tags = metadata["tags"] if "tags" in metadata else []
            title = metadata["catalogue_title"] if "catalogue_title" in metadata else ""
            rows.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": metadata["source"],
                    "pages": metadata["pages"],
                    "embedder": embedder_name,
                    "tags": _tags_json(tags),
                    "catalogue_title": title,
                    "vector": vector,
                }
            )
        return rows

    def _add_rows(self, table, rows: list[dict]) -> None:
        """写入一行批；测试可 monkeypatch 这一层模拟 add 失败。"""
        table.add(rows)

    def _delete_source(self, table, source: str) -> None:
        """按来源名删除行。"""
        table.delete(f"source = {_sql_literal(source)}")

    def count_source_rows(self, source: str) -> int:
        """某来源当前行数；没有表则 0。"""
        if CHUNKS_TABLE not in self.db.table_names():
            return 0
        table = self._open_chunks()
        return table.count_rows(f"source = {_sql_literal(source)}")

    def list_writing_sources(self) -> list[str]:
        """
        仍留在库里的半截入库来源（去掉标记后的正式名）。

        应用场景：问答进程启动时告警，提示对该来源重新 ingest。
        """
        if CHUNKS_TABLE not in self.db.table_names():
            return []
        table = self._open_chunks()
        if table.count_rows() == 0:
            return []
        rows = table.to_lance().to_table(columns=["source"]).to_pylist()
        found = {
            row["source"][: -len(WRITING_MARKER)]
            for row in rows
            if _is_writing_source(row["source"] or "")
        }
        return sorted(found)

    def replace_source(
        self,
        source: str,
        chunks: list[dict],
        vectors: list[list[float]],
        embedder_name: str,
        dim: int,
    ) -> None:
        """
        按来源整批替换写入。

        功能：chunks[i] 与 vectors[i] 对齐。已有表时：先写入暂存来源 → 删旧来源 →
              写入正式行 → 删暂存。暂存 add 失败则旧行不动；正式 add 失败则暂存仍在
              （load 会跳过暂存并告警），不再出现「先删光再失败 → 该来源空库」。
        应用场景：向量层入口写库。
        输入：source — 来源名；chunks — {id, text, metadata}；vectors — 等长向量；
              embedder_name / dim — 与表内已有行必须一致。
        输出：无。
        异常：条数不一致、向量长度不一致或与 dim 不符、维数或 embedder 不一致 → ValueError。
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"块数 {len(chunks)} 与向量数 {len(vectors)} 不一致")
        if _is_writing_source(source):
            raise ValueError(f"来源名不能以 {WRITING_MARKER} 结尾：{source}")
        # 首次建表也要过这一关：以前 create_table 分支不校验 dim，256 维的 hash 抢跑建表后
        # 1024 维的 bge-m3 再也写不进去，只能删库重入（TODO P1·11）
        _check_vectors(vectors, dim)
        rows = self._rows_from_chunks(chunks, vectors, embedder_name)
        names = self.db.table_names()
        if CHUNKS_TABLE not in names:
            self.db.create_table(CHUNKS_TABLE, rows)
            return
        table = self._open_chunks()
        existing_dim = _vector_dim(table)
        if existing_dim != dim:
            raise ValueError(
                f"向量库 dim 不一致：表内 {existing_dim}，本次 {dim}。删掉 {self.db_dir} 后重新入库"
            )
        n_existing = table.count_rows()
        if n_existing:
            probe = table.to_lance().to_table(columns=["embedder"])
            existing_name = probe.to_pylist()[0]["embedder"]
            if existing_name != embedder_name:
                raise ValueError(
                    f"向量库 embedder 不一致：表内 {existing_name}，本次 {embedder_name}"
                )
        staging = _staging_source(source)
        staging_rows = [
            {
                **row,
                "source": staging,
                "id": f"{row['id']}{WRITING_MARKER}",
            }
            for row in rows
        ]
        # 清掉上次崩溃留下的同名暂存，再写入；这一步失败 → 旧正式来源原样保留
        self._delete_source(table, staging)
        self._add_rows(table, staging_rows)
        self._delete_source(table, source)
        self._add_rows(table, rows)
        self._delete_source(table, staging)

    def load_payload(self) -> tuple[list[dict], str, int]:
        """
        读出检索 payload，不含 vector。

        功能：select 正文与标签列；dim 来自 schema。
        应用场景：召回层启动装库、建 BM25。
        输入：无，读 self.db_dir。
        输出：(records, embedder_name, dim)。record 含 id/text/metadata/source/pages；
              metadata 含 tags 与 catalogue_title（无标题则空串）。
        异常：没有表或没有行 → FileNotFoundError；embedder 不一致 → ValueError。
        """
        table = self._open_chunks()
        n_rows = table.count_rows()
        if n_rows == 0:
            raise FileNotFoundError(f"向量表为空，请先入库：{self.db_dir}")
        dim = _vector_dim(table)
        lance_ds = table.to_lance()
        rows = lance_ds.to_table(columns=PAYLOAD_COLUMNS).to_pylist()
        writing = sorted(
            {
                row["source"][: -len(WRITING_MARKER)]
                for row in rows
                if _is_writing_source(row.get("source") or "")
            }
        )
        if writing:
            print(
                "警告：发现半截入库来源 "
                + "、".join(writing)
                + "，检索已跳过暂存行；请对该来源重新 python scripts/ingest/ingest.py"
            )
        embedder_name = None
        records = []
        for row in rows:
            if _is_writing_source(row.get("source") or ""):
                continue
            if embedder_name is None:
                embedder_name = row["embedder"]
            elif row["embedder"] != embedder_name:
                raise ValueError(
                    f"向量库 embedder 不一致：已有 {embedder_name}，行 {row['id']} 是 {row['embedder']}"
                )
            title = row["catalogue_title"] if row["catalogue_title"] else ""
            metadata = {
                "source": row["source"],
                "pages": _pages_list(row["pages"]),
                "tags": _tags_from_cell(row["tags"]),
                "catalogue_title": title,
            }
            records.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": metadata,
                    "source": row["source"],
                    "pages": _pages_list(row["pages"]),
                }
            )
        if not records:
            raise FileNotFoundError(
                f"向量表无可用行（可能只剩半截入库）：{self.db_dir}；"
                "请重新 python scripts/ingest/ingest.py"
            )
        return records, embedder_name, dim

    def search_vectors(
        self,
        query_vector: list[float],
        filters: dict,
        limit: int,
    ) -> list[dict]:
        """
        过滤后按余弦距离近邻。

        功能：distance_type=cosine；来源空串则不加 where（tags 不进 where，它是加权信号，ADR-0035）。
              _distance 是 1-余弦，score 转回余弦。跳过半截入库暂存行。
        应用场景：召回层向量一路。
        输入：query_vector — 查询向量；filters — 只用 filters.source；limit — 条数。
        输出：[{id, kind, text, score, source, pages}, ...]，已按近邻排序。
        """
        table = self._open_chunks()
        query = table.search(query_vector).distance_type("cosine")
        if filters["source"]:
            query = query.where(f"source = {_sql_literal(filters['source'])}")
        # 多取若干条再滤掉暂存，避免半截行占满 limit
        rows = query.limit(limit + 16).to_list()
        hits = []
        for row in rows:
            if _is_writing_source(row.get("source") or ""):
                continue
            hits.append(
                {
                    "id": row["id"],
                    "kind": "chunk",
                    "text": row["text"],
                    "score": 1.0 - float(row["_distance"]),
                    "source": row["source"],
                    "pages": _pages_list(row["pages"]),
                }
            )
            if len(hits) >= limit:
                break
        return hits


def tokenize(text: str) -> list[str]:
    """
    把一段正文切成 BM25 用的词。

    功能：抽出 [A-Za-z0-9]+；连续汉字做成相邻二字，单字成一词。
    应用场景：建倒排与打查询。
    输入：text — 知识块或查询正文。
    输出：小写词列表；空文为 []。
    """
    tokens = [match.group(0).lower() for match in _LATIN_OR_DIGIT.finditer(text or "")]
    # 连续汉字单独攒，避开夹在中间的标点与拉丁词
    buffer = []
    for char in text or "":
        if "\u4e00" <= char <= "\u9fff":
            buffer.append(char)
            continue
        tokens.extend(_han_grams(buffer))
        buffer = []
    tokens.extend(_han_grams(buffer))
    return tokens


def _han_grams(chars: list[str]) -> list[str]:
    """连续汉字 → 二字；只剩一字则保留该字。"""
    if not chars:
        return []
    if len(chars) == 1:
        return chars
    return [chars[index] + chars[index + 1] for index in range(len(chars) - 1)]


class Bm25Index:
    """一批文档的 BM25 倒排。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []
        self.doc_len = []
        self.avg_len = 0.0
        self.df = defaultdict(int)
        self.tf = []
        self.n_docs = 0

    def build(self, documents: list[tuple[str, str]]) -> None:
        """
        用 (id, text) 列表建倒排。

        功能：统计 df / 每篇 tf / 平均长度。
        应用场景：知识库 load 之后。
        输入：documents — [(id, text), ...]，id 与知识块 id 相同。
        输出：无，写入实例属性。
        """
        self.doc_ids = []
        self.doc_len = []
        self.df = defaultdict(int)
        self.tf = []
        lengths = []
        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token in counts:
                self.df[token] += 1
            self.doc_ids.append(doc_id)
            self.tf.append(dict(counts))
            length = len(tokens) or 1
            self.doc_len.append(length)
            lengths.append(length)
        self.n_docs = len(self.doc_ids)
        self.avg_len = (sum(lengths) / self.n_docs) if self.n_docs else 1.0

    def ranked_ids(self, query: str, limit: int) -> list[tuple[str, float]]:
        """
        按 BM25 分数降序取前 limit 条。

        功能：查询无词则 []。
        应用场景：混合检索的词面一路。
        输入：query — 查询原文；limit — 最多条数。
        输出：[(id, score), ...]，score 为 BM25。
        """
        query_tokens = tokenize(query)
        if not query_tokens or self.n_docs == 0:
            return []
        scores = [0.0] * self.n_docs
        seen = set()
        for token in query_tokens:
            if token in seen:
                continue
            seen.add(token)
            df = self.df[token]
            if df == 0:
                continue
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for index, term_freq in enumerate(self.tf):
                tf = term_freq.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[index] / self.avg_len)
                scores[index] += idf * (tf * (self.k1 + 1.0)) / denom
        ranked = [
            (self.doc_ids[index], scores[index])
            for index in range(self.n_docs)
            if scores[index] > 0
        ]
        ranked.sort(key=lambda row: row[1], reverse=True)
        return ranked[:limit]
