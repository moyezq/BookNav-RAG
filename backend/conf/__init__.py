# -*- coding: utf-8 -*-
"""
项目公共配置。

职责：读 backend/conf/config.yaml 的相对目录；读仓库根 .env（第三方地址与密钥）；再导出 backend.conf.base 的层开关。
对外：ROOT（backend 目录）、REPO_ROOT、层与模块常量、SOURCE_DIR、RESULT_DIR、LANCEDB_DIR、QA_DIR、
      EMBED_API_URL、EMBED_API_KEY、CHAT_API_URL、CHAT_MODEL_NAME。
权重目录不在这里：在 service.embed.MODEL_DIR（service/embed/models/）。
密钥边界只有两处——进程环境变量与本仓 `.env`，不读别的仓。
"""

from __future__ import annotations

import os
from pathlib import Path

from backend.conf.base import REPO_ROOT, ROOT
from backend.conf.base import (  # noqa: F401  — 供 from backend.conf import 使用
    CANDIDATE_FLOOR,
    CANDIDATE_MULTIPLIER,
    CHAT_TIMEOUT_SEC,
    CHITCHAT_PROMPT,
    CLASSIFY_PROMPT,
    DEBUG_TAG_MAX_CHARS,
    DEBUG_WRITE_QA,
    EMBED_DIM_BY_NAME,
    EMBED_MODEL_NAME,
    EMBED_REQUEST_BATCH,
    EMBED_REQUEST_TIMEOUT_SEC,
    EXTRACT_CATALOGUE_PREVIEW_PAGES,
    EXTRACT_CATALOGUE_STOP_PAGES,
    HANDBOOK_PROMPT,
    HISTORY_MAX_TURNS,
    SESSION_ID_MAX_CHARS,
    SESSION_TURNS_MAX,
    LLM_MAX_TOKENS,
    MAX_DOC_TOKENS,
    QUERY_TAG_PROMPT,
    REFUSE_MIN_SCORE,
    REFUSE_ON_LEXICAL_FALLBACK,
    REFUSE_TEXT,
    REWRITE_PROMPT,
    RRF_K,
    SKIP_DIR_NAMES,
    SNIPPET_CHARS,
    SOURCE_ALLOWED_SUFFIXES,
    STANDALONE_PROMPT,
    TAG_BOOST_MAX,
    TAG_BOOST_PER_HIT,
    TAG_LEXICAL_MIN_CHARS,
    TOP_K,
    USE_REWRITE,
)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_flat_yaml(path: Path) -> dict[str, str]:
    """读只有「键: 值」的 yaml。不支持嵌套。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到配置 {path}")
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _parse_env_file(path: Path) -> dict[str, str]:
    """读 KEY=VALUE 行。跳过空行和注释。"""
    parsed: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")
    return parsed


def _fill_from_file(path: Path) -> None:
    """只填 os.environ 里仍为空的键。"""
    for key, value in _parse_env_file(path).items():
        if not os.environ.get(key, "").strip():
            os.environ[key] = value


def apply_env() -> None:
    """已有环境变量不动；只读本仓 .env 补空缺。

    这里以前还会读 `ANYURL2TEXT_ENV` 指向的**另一个仓的 .env** 当兜底（抽取委托给 anyurl2text
    时代的共用密钥做法）。抽取解耦后（ADR-0034）那半已无理由，还会把外仓的所有键注进本进程；
    2026-10-10 拆掉，见 TODO P2·25。
    """
    local = REPO_ROOT / ".env"
    if local.is_file():
        _fill_from_file(local)


_raw = _load_flat_yaml(_CONFIG_PATH)
SOURCE_DIR = ROOT / _raw["data_source"]
RESULT_DIR = ROOT / _raw["data_result"]
LANCEDB_DIR = ROOT / _raw["data_lancedb"]
QA_DIR = RESULT_DIR / "qa"

apply_env()

EMBED_API_URL = os.getenv("EMBED_API_URL", "http://127.0.0.1:8030/v1/embeddings")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
CHAT_API_URL = os.getenv("CHAT_API_URL", "https://api.deepseek.com/v1/chat/completions")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "deepseek-flash")

