"""
把一份 URL 或本地文件抽成原文，再按 PIPELINE 走到向量库或 KV 条目后退出。
不问答、不进抽取 entrance、不进检索。线上主路径与 tests 都不 import 本文件。

原理（按系统流程）
  线上完整链路是：源文件 → 抽取层可分块原文 → 向量层知识块 → 提问口检索。
  本脚本是 KV 实验支线（ADR-0005、ADR-0030）：anyurl2text 抽原文；
  PIPELINE=text 接到向量层；PIPELINE=knowledge 抽成 KV 条目写入本地 JSON。
  不进抽取 entrance、不进召回。公开版不保留 KV 实验时，可删本文件。

  一条 knowledge 样本逐步发生什么：
    1. POST anyurl2text /extract 或 /extract/file，拿到 content 与 meta.ext。
    2. 从正文按出现顺序收集 http(s) URL，写下原文。
    3. 按 kind 分块（offline.vector.chunk.split_text_for_kind）。
    4. 每块调 LLM，收成细条目 items（index / content，content 须来自原文）。
    5. 原文里出现过、但 items 尚未覆盖的 URL 补进每一块。
    6. 每块再调 LLM，把细条目重组成更利于一次命中的检索行。
    7. {index, content} → {key, value}，写入 step7_knowledge_kv.json 与 kv_store.json。到此停止。
  一条 text 样本：1～2 步同上；随后写可分块原文并调向量层写入 data/lancedb。不写 KV。

  例子（输入就是下面的常量；换文件或模型会变）
    FILE_URL 为空、LOCAL_FILE 指向 data/source 下一份能被 anyurl2text 抽正文的文件，
    PIPELINE=knowledge。
      期望：data/result/kv/step7_knowledge_kv.json 是非空列表，原文里的 URL 出现在某条 value。
    同一份源、PIPELINE=text。
      期望：data/result/<source>/chunkable_source.json 已写，随后写入 data/lancedb；不写 KV。

怎么跑（本文件入口；编号和上面「一条样本」不是同一套）
  职责：按顶部常量抽一份源，把原理前几步跑完后退出。生产入库走 scripts/ingest/ingest.py。
  入口步骤：
    1. extract_text — 调 anyurl2text，对应原理 1。
    2. write_extracted — 扫 URL、写原文，对应原理 2。
    3. PIPELINE=text：write_url_chunkable_source + run_vector。
       PIPELINE=knowledge：split_chunks → atomic_extract → fill_missing_urls
       → retrieve_rebuild → write_kv（原理 3～7）。
  输入：下面常量 PIPELINE / FILE_URL / LOCAL_FILE。
  前置：anyurl2text 已起；DEEPSEEK_API_KEY 在进程环境或本仓 .env。
        text 入库还需 python -m embed.server。
  产出：data/result/kv/step1_extract_response.json、step2_extracted.txt；
        knowledge 另有 step5_atomic_chunks.json、step6_retrieve_rows.json、
        step7_knowledge_kv.json、kv_store.json；
        text 另有 data/result/<source>/chunkable_source.json 与 data/lancedb。
  对外：extract_text … write_kv；write_url_chunkable_source；call_chat_json。
  跑法：仓库根 uv run python scripts/url_to_knowledge.py
        到点 RUN_LIMIT_SEC 打全部线程栈并退出（本脚本按块调 LLM，默认 5 分钟不够）。
"""

from __future__ import annotations

import faulthandler
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Final, Literal, TypedDict
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import conf  # ROOT 在 conf.base；读 yaml 与 .env
from conf import (
    ANYURL2TEXT_URL,
    CHAT_API_URL,
    CHAT_MODEL_NAME,
    MAX_DOC_TOKENS,
    RESULT_DIR as KB_RESULT_DIR,
    SOURCE_DIR,
)
from offline.extract.enrich import infer_ingest_tags
from offline.store.kv_store import LocalKvStore
from offline.vector.chunk import split_text_for_kind
from offline.vector.entrance import run_vector

RESULT_DIR = KB_RESULT_DIR / "kv"

# ========== 按需修改 ==========
PIPELINE: Final[Literal["text", "knowledge"]] = "knowledge"
FILE_URL: Final[str] = ""  # 有值走 POST /extract
LOCAL_FILE: Final[Path] = SOURCE_DIR / "sample_declaration.txt"  # 改成 data/source 里实际有的文件
DEEPSEEK_API_KEY: Final[str] = ""  # 留空：环境变量 / 本仓 .env
# KV 单块原文更长，不用在线问答的 120s / 2048 tokens
CHAT_TIMEOUT_SEC: Final[int] = 600
LLM_MAX_TOKENS: Final[int] = 8192
EXTRACT_TIMEOUT_SEC: Final[int] = 300
# 抽原文上限 + 若干次 Chat；到点打栈退出
RUN_LIMIT_SEC: Final[int] = 1800
# ==============================

CHAT_MAX_ATTEMPTS: Final[int] = 3
URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://\S+")
DOC_KINDS_IMAGE_INDEX: Final[frozenset[str]] = frozenset({"docx", "pdf", "excel", "xlsx"})

ATOMIC_PROMPT: Final[str] = """
你是知识库结构化专家。你将收到一份从 __KIND__ 提取出的原文（可能包含 <table>... 结构以及 URL 链接）。

目标：把内容转成“原子化知识点二维表”，便于检索与匹配。

要求：
1）先给出文档主题 topic（尽量完整、准确）
2）输出知识点 items：必须尽可能多，覆盖每个章节/表格中的可检索信息；每一行必须尽可能“原子化”
   - index：原子化知识点名（例如：报到时间、比赛时间、比赛地点、报名截止时间、关门时间等）
   - content：尽可能保留原文的对应句子/条目（允许多行拼接，但必须来自原文，不得发明/总结/改写）
3）强约束：content 中不应出现连续重复的文本
4）强约束：下面列出的每一个 URL 都必须在 items 中至少出现 1 次（通常放在 content 中）
5）禁止输出 Markdown，禁止输出多余解释文字，只允许输出一个 JSON 对象
6）输出必须是合法 JSON；字符串中的双引号须转义；换行可保留为 \\n

输出 JSON 结构必须满足：
{
  "topic": "string",
  "items": [
    {
      "index": "string",
      "content": "string"
    }
  ]
}

必须覆盖的 URL 列表：
__URLS__

原文如下（请完整阅读后再输出 JSON）：
__DOC_TEXT__
""".strip()

RETRIEVE_PROMPT: Final[str] = """
你是知识库“检索结构化”专家。

你将收到：
1）某一段原文（chunk_index=__CHUNK_INDEX__，来源：__KIND__）
2）该段对应的“原子化知识点 JSON”（来自上一步抽取，分得很细）

目标：
把该段的原子化知识点重组为更利于检索的二维知识库结构。要求检索时尽量“一次命中得到足够答案”。

重要约束（必须遵守）：
1）输出必须是一个 JSON 对象，禁止输出 Markdown，禁止输出解释文字。
2）每条知识的 content 必须来自“原文片段”的原句/原条目拼接（允许换行拼接、允许重排顺序），不得发明/总结/改写。
3）index 是“可检索的标题/意图”（例如：报名方式、参赛资格、关门时间、注意事项、联系方式等）。
4）尽量减少行数：把同一意图下的内容合并到同一行，但不要把完全无关的内容硬塞在一起。
5）不得输出空 rows；若信息很少，也至少输出 1 条。
6）输出必须是合法 JSON；字符串中的双引号须转义；换行可保留为 \\n

输出 JSON 结构（严格遵守字段名）：
{
  "chunk_index": __CHUNK_INDEX__,
  "rows": [
    {
      "index": "string",
      "content": "string"
    }
  ]
}

原子化知识点 JSON：
__ATOM_ITEMS_JSON__

原文片段如下（请完整阅读后再输出 JSON）：
__DOC_TEXT__
""".strip()


class KnowledgeItem(TypedDict):
    index: str
    content: str


class TextData(TypedDict):
    kind: str
    urls: list[str]
    text: str


class AtomicChunk(TypedDict):
    chunk_index: int
    topic: str
    text: str
    items: list[KnowledgeItem]


class KvPair(TypedDict):
    key: str
    value: str


class ChunkableSource(TypedDict):
    source: str
    kind: str
    text: str
    tags: list[dict]


def _resolve_deepseek_api_key() -> str:
    """常量非空优先，否则进程环境变量（含本仓 .env 补进的值）。"""
    return DEEPSEEK_API_KEY.strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()


def _strip_thinking(text: str) -> str:
    """去掉思维链前缀：思考模式若被打开，DeepSeek 仍可能带 </think>。"""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return (text or "").strip()


def extract_json_payload(text: str) -> dict | list | None:
    """
    从模型输出里截出第一段能 parse 的 JSON 对象或数组。

    功能：容忍前后废话与思维链，只取 JSON。
    应用场景：call_chat_json 解析原子化 / 检索重组的回复。
    输入：text — 模型原始 content。
    输出：dict 或 list；截不出则 None。
    """
    text = _strip_thinking(text or "")
    if not text:
        return None

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        raw = text[obj_start : obj_end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        raw = text[arr_start : arr_end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


def call_chat_json(full_prompt: str, *, stage: str = "") -> dict | list:
    """
    调 DeepSeek Chat Completions，要求回复能解析成 JSON。

    功能：关思考模式、json_object、显式 max_tokens；最多重试 CHAT_MAX_ATTEMPTS 次后截 JSON。
    应用场景：atomic_extract、retrieve_rebuild。
    输入：full_prompt — 整段 user 提示词；stage — 日志标签（atomic_1 / retrieve_1）。
    输出：解析后的 dict 或 list。
    异常：缺 API key、或次数用尽仍失败 → RuntimeError。
    """
    api_key = _resolve_deepseek_api_key()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY：设进程环境变量或写入本仓 .env")

    last_err: Exception | None = None
    for attempt in range(1, CHAT_MAX_ATTEMPTS + 1):
        try:
            started = time.perf_counter()
            response = requests.post(
                CHAT_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": CHAT_MODEL_NAME,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": LLM_MAX_TOKENS,
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                },
                timeout=CHAT_TIMEOUT_SEC,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = extract_json_payload(content)
            elapsed = round(time.perf_counter() - started, 2)
            print(f"模型调用 stage={stage} attempt={attempt} cost={elapsed}s", flush=True)
            if parsed is not None:
                return parsed
            last_err = RuntimeError("模型输出不是合法 JSON")
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            last_err = exc
            print(f"模型调用失败 stage={stage} attempt={attempt}: {exc}", flush=True)
    raise RuntimeError(f"模型调用失败 stage={stage}, last_error={last_err}") from last_err


def normalize_item_fields(index_text: str, content_text: str) -> tuple[str, str]:
    """
    压缩 index 空白、折叠 content 连续空行。

    功能：index 多空白压成单空格；content 连续空行压成最多两行换行。
    应用场景：原子化清洗与检索重组共用。
    输入：index_text — 标题；content_text — 正文。
    输出：(index, content) 规范化后的二元组。
    """
    index_text = re.sub(r"\s+", " ", (index_text or "").strip())
    content_text = (content_text or "").strip()
    content_text = re.sub(r"\n{3,}", "\n\n", content_text)
    return index_text, content_text


def clean_atomic_items(items: list[dict]) -> list[KnowledgeItem]:
    """
    清洗原子化 items：规范化字段，去掉 index 与 content 首行重复。

    功能：空条目丢弃；content 第一行若等于 index 则剥掉。
    应用场景：atomic_extract 收 LLM 结果、fill_missing_urls 补 URL 之后。
    输入：items — [{index, content}, ...]。
    输出：清洗后的同结构列表。
    """
    cleaned: list[KnowledgeItem] = []
    for item in items:
        index_text, content_text = normalize_item_fields(
            str(item.get("index", "")),
            str(item.get("content", "")),
        )
        if index_text and content_text:
            content_lines = content_text.splitlines()
            if content_lines and content_lines[0].strip() == index_text:
                content_text = "\n".join(content_lines[1:]).strip()
        if not index_text and not content_text:
            continue
        cleaned.append({"index": index_text, "content": content_text})
    return cleaned


def write_result(filename: str, text: str) -> Path:
    """
    把本步产物写到 data/result/kv/<filename>。

    功能：确保 kv 目录存在后写 UTF-8 文本。
    应用场景：各步写进磁盘共用。
    输入：filename — 如 step2_extracted.txt；text — 文件内容。
    输出：写出的路径。
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path


def collect_urls_from_text(text: str) -> list[str]:
    """
    从原文里按出现顺序去重收集 http(s) 链接。

    功能：正则扫 https?://，同一 URL 只留第一次。
    应用场景：write_extracted 填 urls；atomic_extract 提示词要覆盖 URL 列表。
    输入：text — 抽取后的原文。
    输出：URL 字符串列表。
    """
    urls: list[str] = []
    seen: set[str] = set()
    for url in URL_PATTERN.findall(text or ""):
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def url_source_name() -> str:
    """
    URL 或本地文件收成来源目录名。

    功能：FILE_URL 用路径最后一段；否则用本地文件 stem。
    应用场景：可分块原文的 source、向量库子目录、KV id 前缀。
    输入：无，读 FILE_URL / LOCAL_FILE。
    输出：不含路径分隔符的字符串。
    """
    file_url = FILE_URL.strip()
    if file_url:
        parsed = urlparse(file_url)
        name = Path(parsed.path).stem or parsed.netloc or "url-source"
        slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", name).strip("-")
        return slug or "url-source"
    return Path(LOCAL_FILE).stem


def extract_text() -> dict:
    """
    调 anyurl2text 抽出原文。

    功能：FILE_URL 非空则 POST /extract，否则把 LOCAL_FILE 以 multipart 传到 /extract/file。
    应用场景：流水线第 1 步。
    输入：读模块常量 FILE_URL / LOCAL_FILE / ANYURL2TEXT_URL。
    输出：anyurl2text 的完整 JSON（含 content、meta、error）。
    异常：HTTP 失败、error 非空、content 为空、meta.ext 缺失 → RuntimeError。
    """
    file_url = FILE_URL.strip()
    if file_url:
        print(f"[1] POST {ANYURL2TEXT_URL}/extract url={file_url}", flush=True)
        response = requests.post(
            f"{ANYURL2TEXT_URL}/extract",
            json={"url": file_url},
            timeout=EXTRACT_TIMEOUT_SEC,
        )
    else:
        local_path = Path(LOCAL_FILE)
        if not local_path.is_file():
            raise RuntimeError(f"本地文件不存在: {local_path}")
        print(f"[1] POST {ANYURL2TEXT_URL}/extract/file file={local_path}", flush=True)
        with local_path.open("rb") as file_obj:
            response = requests.post(
                f"{ANYURL2TEXT_URL}/extract/file",
                files={"file": (local_path.name, file_obj)},
                timeout=EXTRACT_TIMEOUT_SEC,
            )

    response.raise_for_status()
    payload = response.json()
    error = payload["error"]
    if error:
        raise RuntimeError(f"anyurl2text 抽取失败: {error}")
    content = payload["content"]
    if not str(content).strip():
        raise RuntimeError("anyurl2text 返回的 content 为空")
    ext = payload["meta"]["ext"]
    if not ext:
        raise RuntimeError("anyurl2text 未给出 meta.ext")
    print(f"[1] ok ext={ext} chars={len(content)}", flush=True)
    extract_path = write_result(
        "step1_extract_response.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    print(f"[1] 已写入 {extract_path}", flush=True)
    return payload


def write_extracted(extract_payload: dict) -> TextData:
    """
    从抽取结果整理 kind / urls / text 并写进磁盘。

    功能：kind 取 meta.ext；urls 从正文正则扫描；写 step2_extracted.txt。
    应用场景：流水线第 2 步。PIPELINE=text 时接着写可分块原文并入库。
    输入：extract_payload — extract_text 的响应。
    输出：{kind, urls, text}，给后续分块与知识库步骤。
    """
    text = extract_payload["content"]
    kind = extract_payload["meta"]["ext"]
    urls = collect_urls_from_text(text)
    text_data: TextData = {"kind": kind, "urls": urls, "text": text}
    extracted_path = write_result("step2_extracted.txt", text)
    print(
        f"[2] 已写入 {extracted_path} kind={kind} chars={len(text)} urls={len(urls)}",
        flush=True,
    )
    return text_data


def write_url_chunkable_source(text_data: TextData) -> ChunkableSource:
    """
    URL 适配器吐可分块原文（无页，用 text）。

    功能：写 data/result/<source>/chunkable_source.json。
    应用场景：PIPELINE=text，write_extracted 之后。
    输入：text_data — write_extracted 的 {kind, text, urls}。
    输出：可分块原文对象。
    """
    source_name = url_source_name()
    tags = infer_ingest_tags(text_data["text"], source_name)
    payload: ChunkableSource = {
        "source": source_name,
        "kind": text_data["kind"],
        "text": text_data["text"],
        "tags": tags,
    }
    source_dir = KB_RESULT_DIR / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / "chunkable_source.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3] 可分块原文 已写入 {path}", flush=True)
    return payload


def split_chunks(text_data: TextData) -> list[str]:
    """
    按文件类型把原文切成原子化用的块。

    功能：调用 offline.vector.chunk.split_text_for_kind；pdf 按页组块，其余按段。空块视为失败。
    应用场景：流水线第 3 步，只在 PIPELINE=knowledge 时调用。
    输入：text_data — write_extracted 的 {kind, text, urls}。
    输出：块文本列表。
    异常：分块为空 → ValueError。
    """
    # 切分实现在 offline/vector/chunk.py，本步只选 kind 并检查非空
    chunks = split_text_for_kind(text_data["text"], text_data["kind"], max_doc_tokens=MAX_DOC_TOKENS)
    if not chunks:
        raise ValueError("原文分块为空")
    print(f"[3] 分块完成 kind={text_data['kind']} chunks={len(chunks)}", flush=True)
    return chunks


def atomic_extract(chunks: list[str], text_data: TextData) -> list[AtomicChunk]:
    """
    对每一块调用原子化提示词，得到 topic + items。

    功能：按块调 LLM，清洗 items；本步不补 URL（留给 fill_missing_urls）。
    应用场景：流水线第 4 步。
    输入：chunks — split_chunks 的块；text_data — 提供 kind 与 urls。
    输出：[{chunk_index, topic, text, items}, ...]。
    异常：LLM 没返回对象 → RuntimeError。
    """
    kind = text_data["kind"]
    urls_text = "\n".join(text_data["urls"])
    atomic_chunks: list[AtomicChunk] = []
    total = len(chunks)
    for chunk_index, chunk_text in enumerate(chunks, start=1):
        print(f"[4] [{chunk_index}/{total}] 原子化 chars={len(chunk_text)}", flush=True)
        prompt = (
            ATOMIC_PROMPT.replace("__KIND__", kind)
            .replace("__URLS__", urls_text)
            .replace("__DOC_TEXT__", chunk_text)
            .strip()
        )
        data = call_chat_json(prompt, stage=f"atomic_{chunk_index}")
        if not isinstance(data, dict):
            raise RuntimeError(f"原子化输出应为对象，实际为 {type(data).__name__}")
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        items = [item for item in items if isinstance(item, dict)]
        atomic_chunks.append(
            {
                "chunk_index": chunk_index,
                "topic": str(data.get("topic", "") or ""),
                "text": chunk_text,
                "items": clean_atomic_items(items),
            }
        )
    print(f"[4] 原子化完成 chunks={total}", flush=True)
    return atomic_chunks


def fill_missing_urls(atomic_chunks: list[AtomicChunk], text_data: TextData) -> list[AtomicChunk]:
    """
    把原文里出现过、但 items 尚未覆盖的 URL 补进每一块。

    功能：用 URL 所在行的前两行当 index 上下文；默认 index 对文档类用「图片」，其余用「链接」。
          每个 chunk 都会检查全文 URL。
    应用场景：流水线第 5 步。
    输入：atomic_chunks — atomic_extract 产出；text_data — 提供 text 与 kind。
    输出：补完 URL 并再次清洗 items 的同一列表（原地修改后返回）。
    """
    kind = text_data["kind"]
    doc_text = text_data["text"]
    default_index = "图片" if kind in DOC_KINDS_IMAGE_INDEX else "链接"
    doc_urls = URL_PATTERN.findall(doc_text or "")
    doc_lines = [line.rstrip() for line in (doc_text or "").splitlines()]
    filled = 0
    for chunk in atomic_chunks:
        items: list[dict] = list(chunk.get("items", []))
        for url in doc_urls:
            if any(url in str(item.get("index", "")) or url in str(item.get("content", "")) for item in items):
                continue
            ctx_index = default_index
            ctx_lines: list[str] = []
            for index, line in enumerate(doc_lines):
                if url not in line:
                    continue
                prev1 = doc_lines[index - 1].strip() if index - 1 >= 0 else ""
                prev2 = doc_lines[index - 2].strip() if index - 2 >= 0 else ""
                if prev1:
                    ctx_index = prev1
                elif prev2:
                    ctx_index = prev2
                ctx_lines = [part for part in [prev2, prev1, url] if part]
                break
            items.append({"index": ctx_index, "content": "\n".join(ctx_lines) if ctx_lines else url})
            filled += 1
        chunk["items"] = clean_atomic_items(items)
    chunks_path = write_result(
        "step5_atomic_chunks.json",
        json.dumps(atomic_chunks, ensure_ascii=False, indent=2),
    )
    print(f"[5] 已写入 {chunks_path} filled={filled} doc_urls={len(doc_urls)}", flush=True)
    return atomic_chunks


def retrieve_rebuild(atomic_chunks: list[AtomicChunk], kind: str) -> list[KnowledgeItem]:
    """
    把每块的原子化 items 重组为更利于一次命中的检索行。

    功能：按块调检索重组提示词，规范化为 [{index, content}, ...]。
    应用场景：流水线第 6 步。
    输入：atomic_chunks — fill_missing_urls 产出；kind — 文件类型，写入提示词。
    输出：所有块的 rows 拼成一条列表。
    """
    rows: list[KnowledgeItem] = []
    total = len(atomic_chunks)
    for chunk in atomic_chunks:
        chunk_index = int(chunk.get("chunk_index") or 0)
        chunk_text = str(chunk.get("text") or "")
        atom_items = chunk.get("items", [])
        topic = str(chunk.get("topic") or "")
        print(f"[6] [{chunk_index}/{total}] 检索重组 atom_items={len(atom_items)}", flush=True)
        atom_items_text = json.dumps({"topic": topic, "items": atom_items}, ensure_ascii=False, indent=2)
        prompt = (
            RETRIEVE_PROMPT.replace("__KIND__", kind)
            .replace("__CHUNK_INDEX__", str(chunk_index))
            .replace("__ATOM_ITEMS_JSON__", atom_items_text)
            .replace("__DOC_TEXT__", chunk_text)
            .strip()
        )
        data = call_chat_json(prompt, stage=f"retrieve_{chunk_index}")
        if isinstance(data, list):
            raw_rows = data
        elif isinstance(data, dict):
            raw_rows = data.get("rows", [])
        else:
            raw_rows = []
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            index_text, content_text = normalize_item_fields(
                str(row.get("index", "")),
                str(row.get("content", "")),
            )
            if index_text and content_text:
                rows.append({"index": index_text, "content": content_text})
    rows_path = write_result(
        "step6_retrieve_rows.json",
        json.dumps(rows, ensure_ascii=False, indent=2),
    )
    print(f"[6] 已写入 {rows_path} rows={len(rows)}", flush=True)
    return rows


def write_kv(retrieve_rows: list[KnowledgeItem]) -> list[KvPair]:
    """
    把重组行转成 KV 条目并写进磁盘。

    功能：{index, content} → {key, value}；写 step7_knowledge_kv.json，并 upsert 进 kv_store.json。
    应用场景：流水线第 7 步。
    输入：retrieve_rows — retrieve_rebuild。
    输出：kv_list。
    """
    kv_list: list[KvPair] = []
    for row in retrieve_rows:
        key = str(row.get("index", "")).strip()
        value = str(row.get("content", "")).strip()
        if not key and not value:
            continue
        kv_list.append({"key": key, "value": value})

    kv_path = write_result(
        "step7_knowledge_kv.json",
        json.dumps(kv_list, ensure_ascii=False, indent=2),
    )
    print(f"[7] 已写入 {kv_path} items={len(kv_list)}", flush=True)

    source_name = url_source_name()
    store_path = RESULT_DIR / "kv_store.json"
    store = LocalKvStore()
    if store_path.is_file():
        store = LocalKvStore.load(store_path)
    entries = []
    for index, item in enumerate(kv_list, start=1):
        entries.append(
            {
                "id": f"{source_name}-kv-{index:04d}",
                "key": item["key"],
                "value": item["value"],
                "metadata": {"source": source_name, "tags": []},
            }
        )
    store.upsert(entries)
    store.save(store_path)
    print(f"[7] KV 存储 已写入 {store_path} records={len(store.records)}", flush=True)
    return kv_list


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    source = FILE_URL.strip() or str(LOCAL_FILE)
    print(f"PIPELINE={PIPELINE}", flush=True)
    print(f"source={source}", flush=True)

    # 1. 抽原文：FILE_URL 或 LOCAL_FILE → extract_payload
    extract_payload = extract_text()
    # 2. 扫 URL、写原文 → data/result/kv/step2_extracted.txt
    text_data = write_extracted(extract_payload)
    if PIPELINE == "text":
        # 3. 可分块原文 → 向量层写入 data/lancedb
        source_obj = write_url_chunkable_source(text_data)
        run_vector(KB_RESULT_DIR / source_obj["source"] / "chunkable_source.json")
        print(f"完成 text kind={text_data['kind']} source={source_obj['source']}", flush=True)
    else:
        # 3. 按 kind 分块（offline.vector.chunk.split_text_for_kind）
        chunks = split_chunks(text_data)
        # 4. LLM 原子化
        atomic_chunks = atomic_extract(chunks, text_data)
        # 5. 补漏 URL → data/result/kv/step5_atomic_chunks.json
        atomic_chunks = fill_missing_urls(atomic_chunks, text_data)
        # 6. LLM 检索重组 → data/result/kv/step6_retrieve_rows.json
        retrieve_rows = retrieve_rebuild(atomic_chunks, text_data["kind"])
        # 7. KV 条目 → data/result/kv/step7_knowledge_kv.json
        kv_list = write_kv(retrieve_rows)
        print(f"完成 knowledge items={len(kv_list)}", flush=True)
