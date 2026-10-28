# -*- coding: utf-8 -*-
"""
向量层入口：可分块原文 → 知识块 → 调向量模型服务 → LanceDB。

职责：读 backend/data/result/<source>/chunkable_source.json，写 step4_chunks.json，按来源写入 backend/data/lancedb 的 chunks 表。
      不问答。不实现模型、不下权重。不写 step5 JSON。
前置：抽取层已写出可分块原文；EMBEDDER_KIND=http 时 python -m service.embed.server 已起。
      由 scripts/ingest/ingest.py 调用；本文件无 __main__。
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.conf import (
    EMBED_API_KEY,
    EMBED_API_URL,
    EMBED_DIM,
    EMBED_DIM_BY_NAME,
    EMBED_MODEL_NAME,
    EMBED_REQUEST_BATCH,
    EMBED_REQUEST_TIMEOUT_SEC,
    EMBEDDER_KIND,
    LANCEDB_DIR,
    MAX_DOC_TOKENS,
    RESULT_DIR,
    SKIP_DIR_NAMES,
)
from backend.offline.vector.chunk import chunks_from_chunkable_source
from backend.offline.vector.store import HashEmbedder, HttpEmbedder, LanceVectorStore


def list_chunkable_paths() -> list[Path]:
    """
    列出各来源已写出的可分块原文。

    功能：扫 backend/data/result/<source>/chunkable_source.json，跳过 kv / qa。
    应用场景：向量层入口。
    输入：无，读 RESULT_DIR。
    输出：按路径排序的 JSON 路径列表。
    异常：一个都没有 → FileNotFoundError。
    """
    paths = []
    if not RESULT_DIR.is_dir():
        raise FileNotFoundError(f"未找到产出目录 {RESULT_DIR}")
    for folder in sorted(RESULT_DIR.iterdir()):
        if not folder.is_dir() or folder.name in SKIP_DIR_NAMES:
            continue
        source_path = folder / "chunkable_source.json"
        if source_path.is_file():
            paths.append(source_path)
    if not paths:
        raise FileNotFoundError(f"未找到可分块原文，先跑 python scripts/ingest/ingest.py")
    return paths


def make_embedder():
    """按 EMBEDDER_KIND 选 Hash 或 HTTP 客户端。"""
    if EMBEDDER_KIND == "http":
        return HttpEmbedder(
            EMBED_API_URL,
            EMBED_MODEL_NAME,
            EMBED_API_KEY,
            timeout_sec=EMBED_REQUEST_TIMEOUT_SEC,
            request_batch=EMBED_REQUEST_BATCH,
        )
    return HashEmbedder(dim=EMBED_DIM)


def run_vector(source_path: Path) -> Path:
    """
    一份可分块原文走完分块、向量、写成文件。

    功能：分块 → 调向量服务 → 按来源替换写入 LanceDB chunks 表 → 写 step4_chunks.json。
          step4 在写库成功之后才写成文件：embed 或写库失败时不许留下「step4 新 / 库旧」的半成品。
    应用场景：scripts/ingest/ingest.py 阶段2。
    输入：source_path — chunkable_source.json。
    输出：LanceDB 目录路径。
    异常：知识块为空 → ValueError。
    """
    result_dir = source_path.parent
    source_obj = json.loads(source_path.read_text(encoding="utf-8"))
    chunks = chunks_from_chunkable_source(source_obj, max_doc_tokens=MAX_DOC_TOKENS)
    if not chunks:
        raise ValueError(f"知识块为空，检查可分块原文 {source_path}")
    print(f"[向量层] 分块完成 chunks={len(chunks)}（step4 等写库成功后才写文件）")

    embedder = make_embedder()
    print(f"[向量层] embedder={embedder.name} source={source_obj['source']}")
    if EMBEDDER_KIND != "http":
        # 临时占位向量不是语义检索：它建出的表写死在 EMBED_DIM 维，换回语义模型时对不上（TODO P1·11）
        print(f"[向量层] 注意：EMBEDDER_KIND={EMBEDDER_KIND}，本次入库用的是临时占位向量，不是语义模型")
    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.embed_texts(texts)
    dim = embedder.dim or len(vectors[0])
    expected_dim = EMBED_DIM_BY_NAME.get(embedder.name)
    if expected_dim is not None and expected_dim != dim:
        raise ValueError(
            f"embedder={embedder.name} 期望 {expected_dim} 维，实际 {dim} 维："
            f"核对 backend/conf/base.py 的 EMBED_DIM / EMBED_MODEL_NAME（EMBED_DIM_BY_NAME），"
            f"以及 embed 服务返回的维数"
        )
    store = LanceVectorStore(LANCEDB_DIR)
    store.replace_source(source_obj["source"], chunks, vectors, embedder.name, dim)
    print(f"[向量层] 已写入 {LANCEDB_DIR} source={source_obj['source']} chunks={len(chunks)} dim={dim}")

    # 写库成功才写 step4 文件：失败时库里还是旧的，step4 不许单独变成新的
    chunks_path = result_dir / "step4_chunks.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[向量层] 已写入 {chunks_path} chunks={len(chunks)}")
    return LANCEDB_DIR
