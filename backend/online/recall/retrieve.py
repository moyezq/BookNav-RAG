# -*- coding: utf-8 -*-
"""
召回层：LanceDB chunks 表看成一份知识库。

使用前注意事项：
- 知识块走向量 + BM25 融合。
- 表内 embedder 必须一致。dim 读 schema，不把 vector 列装进内存。
- 旧记录若缺 tags 则当 []：有标签过滤时整条丢掉。
- catalogue_title 空串表示无印刷目录标题，同节邻块跳过。
- 向量服务不可用时退化为纯 BM25（词面兜底）：命中带 lexical_fallback=True，拒答改看词面。
"""

from __future__ import annotations

import json

import requests
from pathlib import Path

from backend.conf import (
    CANDIDATE_FLOOR,
    CANDIDATE_MULTIPLIER,
    DEBUG_WRITE_QA,
    EMBED_API_KEY,
    EMBED_API_URL,
    EMBED_REQUEST_BATCH,
    LANCEDB_DIR,
    MAX_DOC_TOKENS,
    QA_DIR,
    RRF_K,
    TAG_BOOST_MAX,
    TAG_BOOST_PER_HIT,
    TOP_K,
)
from backend.offline.vector.chunk import estimate_tokens_by_chars
from backend.offline.vector.store import Bm25Index, HashEmbedder, HttpEmbedder, LanceVectorStore
from backend.online.filter.tags import empty_filters, source_matches, tag_match_count

# 向量服务不可用时的异常集合；命中后走纯 BM25 兜底
EMBED_UNAVAILABLE = (requests.RequestException, KeyError, ValueError)


def _hit_from_record(record: dict, kind: str, score: float) -> dict:
    """知识块记录收成检索命中。"""
    return {
        "id": record["id"],
        "kind": kind,
        "text": record["text"],
        "score": score,
        "source": record["source"],
        "pages": record["pages"],
    }


def _rrf_fuse(
    vector_ranked: list[dict],
    bm25_ranked: list[dict],
    top_k: int,
    boost_by_id: dict | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion。两路都空则 []。

    功能：score = Σ 1/(k+rank)，再加过滤标签加权（命中 N 条标签 → min(N × TAG_BOOST_PER_HIT, TAG_BOOST_MAX)）。
         标签是软信号，不加权的块不会因为没命中标签被丢掉（ADR-0035）。
         命中对象取先出现的那路，再写融合分。向量路过的 id 另带 vector_score（余弦），供拒答阈值，不进生成字段约定。
    输入：vector_ranked / bm25_ranked — 已按各路分数降序的命中；top_k — 候选窗口截断；
          boost_by_id — {id: 命中标签条数}，空则不加权。
    输出：融合后的检索命中列表。
    """
    fused = {}
    by_id = {}
    vector_cosine = {}
    for rank, hit in enumerate(vector_ranked, start=1):
        fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank)
        by_id[hit["id"]] = hit
        vector_cosine[hit["id"]] = hit["score"]
    for rank, hit in enumerate(bm25_ranked, start=1):
        fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank)
        if hit["id"] not in by_id:
            by_id[hit["id"]] = hit
    for doc_id, matched in (boost_by_id or {}).items():
        if doc_id in fused and matched > 0:
            fused[doc_id] += min(TAG_BOOST_MAX, TAG_BOOST_PER_HIT * matched)
    ordered_ids = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)
    results = []
    for doc_id in ordered_ids[:top_k]:
        hit = dict(by_id[doc_id])
        hit["score"] = fused[doc_id]
        if doc_id in vector_cosine:
            hit["vector_score"] = vector_cosine[doc_id]
        results.append(hit)
    return results


def _merge_catalogue_neighbors(siblings: list[dict], center_id: str, max_tokens: int) -> dict:
    """
    以命中块为中心，把同一目录标题下的邻块并进正文。

    功能：先向后、再向前，直到粗估 token 超过上限。页码取并集。
    应用场景：重排截断后的同节邻块。
    输入：siblings — 同来源同标题、已按 id 排好；center_id — 命中块 id；max_tokens — 上限。
    输出：{text, pages}。
    """
    index = 0
    for position, record in enumerate(siblings):
        if record["id"] == center_id:
            index = position
            break
    used = [siblings[index]]
    tokens = estimate_tokens_by_chars(siblings[index]["text"])
    # 先收后面的续段，再收前面的前提
    for record in siblings[index + 1 :]:
        extra = estimate_tokens_by_chars(record["text"])
        if tokens + extra > max_tokens:
            break
        used.append(record)
        tokens += extra
    for record in reversed(siblings[:index]):
        extra = estimate_tokens_by_chars(record["text"])
        if tokens + extra > max_tokens:
            break
        used.insert(0, record)
        tokens += extra
    pages = []
    seen = set()
    for record in used:
        for page in record["pages"]:
            if page in seen:
                continue
            seen.add(page)
            pages.append(page)
    pages.sort()
    return {
        "text": "\n\n".join(record["text"] for record in used),
        "pages": pages,
    }


class KnowledgeBase:
    """一份逻辑知识库：backend/data/lancedb 的 chunks 表。"""

    def __init__(self, lancedb_dir: Path):
        self.lancedb_dir = lancedb_dir
        self.vector_store = LanceVectorStore(lancedb_dir)
        self.embedder_name = ""
        self.dim = 0
        self.records = []
        self.bm25 = Bm25Index()

    def load(self) -> None:
        """
        读入 chunks 表的 payload，并建 BM25。

        功能：不读 vector 列；用正文建倒排。
        应用场景：打开知识库。
        输入：无，读 self.lancedb_dir。
        输出：无，写入实例属性。
        异常：没有表、表空、或 embedder 不一致 → FileNotFoundError / ValueError。
        """
        records, embedder_name, dim = self.vector_store.load_payload()
        if embedder_name == "hash":
            print("警告 embedder=hash，不是语义检索。请用 EMBEDDER_KIND=http 重新入库")

        self.embedder_name = embedder_name
        self.dim = dim
        self.records = records
        self.bm25.build([(record["id"], record["text"]) for record in records])
        print(f"知识库 records={len(records)} embedder={embedder_name} dim={dim}")

    def tag_catalog(self) -> dict:
        """
        库内已出现的来源与过滤标签对。

        功能：扫描 records 的 source 与 metadata.tags，去重排序。
        应用场景：意图识别给出本轮候选标签。
        输入：无，读 self.records。
        输出：{sources, tags}；tags 为 [{field, value}, ...]。
        """
        sources = set()
        pairs = []
        seen = set()
        for record in self.records:
            if record["source"]:
                sources.add(record["source"])
            metadata = record["metadata"]
            if "tags" not in metadata:
                continue
            for tag in metadata["tags"]:
                key = (tag["field"], tag["value"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"field": tag["field"], "value": tag["value"]})
        pairs.sort(key=lambda tag: (tag["field"], tag["value"]))
        return {
            "sources": sorted(sources),
            "tags": pairs,
        }

    def search(
        self,
        query_vector: list[float],
        filters: dict,
        top_k: int,
        query_text: str = "",
    ) -> list[dict]:
        """
        过滤后混合检索，返回检索命中。

        功能：来源强制筛选；LanceDB 近邻 + BM25 再 RRF，最后按命中标签条数加权（tags 不做强制筛选，ADR-0035）。
        应用场景：召回层混合检索。
        输入：query_vector — 查询向量；filters — source 强制筛选 + tags 加权，空则该维不限；
              top_k — 生成窗口，用来算候选条数；
              query_text — 查询原文，给 BM25。
        输出：[{id, kind, text, score, source, pages}, ...]，按加权后的融合分降序。
              混合检索截到候选窗口（默认 20），不截 TOP_K。向量路过的命中另有 vector_score。
        """
        if len(query_vector) != self.dim:
            raise ValueError(f"查询向量维数 {len(query_vector)} 与库 dim={self.dim} 不一致")

        candidate_n = max(top_k * CANDIDATE_MULTIPLIER, CANDIDATE_FLOOR)
        filtered = [
            record for record in self.records if source_matches(record["source"], filters)
        ]
        if not filtered:
            print(f"来源过滤后 records=0 filters={filters}")
            return []
        allowed_ids = {record["id"] for record in filtered}
        by_id = {record["id"]: record for record in filtered}
        boost_by_id = {
            record["id"]: tag_match_count(record["metadata"], filters["tags"])
            for record in filtered
        }
        vector_ranked = [
            hit
            for hit in self.vector_store.search_vectors(query_vector, filters, candidate_n)
            if hit["id"] in allowed_ids
        ]
        bm25_ranked = []
        for doc_id, score in self.bm25.ranked_ids(query_text, self.bm25.n_docs):
            if doc_id not in allowed_ids:
                continue
            bm25_ranked.append(_hit_from_record(by_id[doc_id], "chunk", score))
            if len(bm25_ranked) >= candidate_n:
                break

        fused = _rrf_fuse(vector_ranked, bm25_ranked, candidate_n, boost_by_id)
        boosted = sum(1 for matched in boost_by_id.values() if matched)
        print(
            f"来源过滤后 records={len(filtered)} filters={filters} 命中标签的候选={boosted} "
            f"vector={len(vector_ranked)} bm25={len(bm25_ranked)} "
            f"fused={len(fused)} candidate_n={candidate_n}"
        )
        return fused

    def retrieve_hits_by_bm25(self, query_text: str, filters: dict, top_k: int) -> list[dict]:
        """
        只走 BM25 的词面召回（向量服务不可用时的兜底）。

        功能：先用 source_matches 收窄，再用同一份倒排打查询，命中标签的按同一套加权抬分（ADR-0035）；
              命中带 lexical_fallback=True。不需要查询向量，所以 embed 服务挂了也能用。
        应用场景：retrieve_hits 里 embedding 失败时。
        输入：query_text — 查询原文；filters — source 强制筛选 + tags 加权，空则该维不限；top_k — 生成窗口。
        输出：同上形状的命中列表，条数 ≤ 候选窗口，按加权后的 BM25 分降序。
              与融合路一致：先在词面窗口里截候选，再按标签加权重排。
        """
        candidate_n = max(top_k * CANDIDATE_MULTIPLIER, CANDIDATE_FLOOR)
        filtered = [
            record for record in self.records if source_matches(record["source"], filters)
        ]
        if not filtered:
            print(f"词面兜底：来源过滤后 records=0 filters={filters}")
            return []
        allowed_ids = {record["id"] for record in filtered}
        by_id = {record["id"]: record for record in filtered}
        hits = []
        for doc_id, score in self.bm25.ranked_ids(query_text, self.bm25.n_docs):
            if doc_id not in allowed_ids:
                continue
            hit = _hit_from_record(by_id[doc_id], "chunk", score)
            matched = tag_match_count(by_id[doc_id]["metadata"], filters["tags"])
            hit["score"] += min(TAG_BOOST_MAX, TAG_BOOST_PER_HIT * matched)
            hit["lexical_fallback"] = True
            hits.append(hit)
            if len(hits) >= candidate_n:
                break
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        print(
            f"词面兜底 records={len(filtered)} filters={filters} bm25={len(hits)} "
            f"candidate_n={candidate_n}"
        )
        return hits

    def expand_hits_by_catalogue(self, hits: list[dict]) -> list[dict]:
        """
        重排后按目录标题把同节邻块并进命中正文。

        功能：空标题不并；同一标题只保留窗口里最先出现的那条，正文换成邻块拼接。
             不改命中字段名。
        应用场景：重排节点截 TOP_K 之后、拒答与组提示之前；评测口对齐生成窗口。
        输入：hits — 已截到 TOP_K 的检索命中。
        输出：同形状命中，条数可能更少（同标题去重）。
        """
        by_id = {record["id"]: record for record in self.records}
        grouped = {}
        for record in self.records:
            title = record["metadata"]["catalogue_title"]
            if not title:
                continue
            key = (record["source"], title)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
        for key in grouped:
            grouped[key].sort(key=lambda row: row["id"])

        seen_titles = set()
        expanded = []
        for hit in hits:
            if hit["kind"] != "chunk" or hit["id"] not in by_id:
                expanded.append(hit)
                continue
            record = by_id[hit["id"]]
            title = record["metadata"]["catalogue_title"]
            if not title:
                expanded.append(hit)
                continue
            key = (hit["source"], title)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            merged = _merge_catalogue_neighbors(
                grouped[key],
                hit["id"],
                MAX_DOC_TOKENS,
            )
            row = dict(hit)
            row["text"] = merged["text"]
            row["pages"] = merged["pages"]
            expanded.append(row)
        print(f"同节邻块 in={len(hits)} out={len(expanded)}")
        return expanded


def _build_embedder(embedder_name: str, dim: int):
    """按库存名字选 Hash 或 HTTP。hash 不是语义模型。"""
    if embedder_name == "hash":
        return HashEmbedder(dim=dim)
    return HttpEmbedder(EMBED_API_URL, embedder_name, EMBED_API_KEY, request_batch=EMBED_REQUEST_BATCH)


def retrieve_hits(
    knowledge_base: KnowledgeBase,
    question: str,
    filters: dict | None = None,
    write_debug: bool = DEBUG_WRITE_QA,
    debug_name: str = "step2_hits.json",
) -> list[dict]:
    """
    把查询编成向量并取检索命中。

    功能：过滤 + 混合检索到候选窗口；write_debug 时写调试题材。空列表交给生成层拒答。
    应用场景：在线图 recall 节点；评测口与测试先拿候选再走 rerank_hits。
    输入：knowledge_base — 已 load；question — 查询 text；filters — 空则不过滤；
          write_debug — 是否写调试题材，默认取 conf 的 DEBUG_WRITE_QA（默认关，
          响应路径不碰磁盘：每问写 141KB 曾把写盘失败变成 HTTP 500）；
          debug_name — 调试题材文件名，调用方按 session 组装，避免并发两问互相覆盖（TODO P2·31）。
    输出：检索命中列表（混合检索约 candidate_n 条，不是 TOP_K）。
    """
    resolved = filters if filters is not None else empty_filters()
    print(f"[阶段2] question={question!r} filters={resolved} top_k={TOP_K}（召回截候选窗口）")
    embedder = _build_embedder(knowledge_base.embedder_name, knowledge_base.dim)
    try:
        query_vector = embedder.embed_texts([question])[0]
        hits = knowledge_base.search(query_vector, resolved, TOP_K, query_text=question)
    except EMBED_UNAVAILABLE as exc:
        # 向量服务不可用 → 退化为纯 BM25。命中带 lexical_fallback，拒答改看词面。
        print(f"向量服务不可用（{type(exc).__name__}: {exc}），退化为纯 BM25 词面召回")
        hits = knowledge_base.retrieve_hits_by_bm25(question, resolved, TOP_K)
    if write_debug:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = QA_DIR / debug_name
        out_path.write_text(
            json.dumps(
                {"question": question, "filters": resolved, "hits": hits},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[阶段2] 已写入 {out_path} hits={len(hits)}")
    for index, hit in enumerate(hits, start=1):
        vector_part = ""
        if "vector_score" in hit:
            vector_part = f" vector={hit['vector_score']:.4f}"
        print(
            f"  [{index}] score={hit['score']:.4f}{vector_part} "
            f"source={hit['source']} pages={hit['pages']} id={hit['id']}"
        )
    return hits


def open_knowledge_base() -> KnowledgeBase:
    """
    打开一份逻辑知识库。

    功能：扫描 LanceDB chunks 表。
    应用场景：python -m backend.online.server 启动；命令行调试与评测口。
    输入：无，读 LANCEDB_DIR。
    输出：已 load 的 KnowledgeBase。
    """
    print("[阶段1] 加载向量库")
    knowledge_base = KnowledgeBase(LANCEDB_DIR)
    knowledge_base.load()
    return knowledge_base
