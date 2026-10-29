"""
入库产物验收：只读检查一份来源的抽取产出与分块产出，回答「这一轮入库有没有坏」。不写库，不调模型。

职责：读 backend/data/result/<来源>/chunkable_source.json、step4_chunks.json 与 backend/data/lancedb 的 chunks 表，
      算出一组可核对的数字，对机械可证的不变量做断言并给出非零退出码。不写库，不调模型。
      让已经发生的数据损坏从「靠人手动翻 JSON」变成「一条命令报出来」。
      丢页不只报个数，还给每个丢页贴原因标签——「按设计丢的导航页 / 空页」
      与「真丢正文」必须分开算，否则正确丢掉的章扉页和目录截断丢的正文长得一模一样。

原理（按系统流程）
  离线流程是：源 → extract_pdf.py（原始页）→ ingest.py（目录区间 + tags → 可分块原文）→ 向量层（分块 → 向量 → LanceDB）。
  坏点位置决定哪个文件能发现它，本脚本只查前两个文件能发现的部分：

  1. 抽取层产出 chunkable_source.json：pages 是源的真实页。查「源一共几页」。
  2. 分块产出 step4_chunks.json：每个知识块带 metadata.pages（声明页码）、metadata.part、metadata.catalogue_title。
     - **页覆盖**：块声明页的并集 vs 源全部页。差集就是「一个字都没进任何块」的源页。
       这是静默故障：抽出来的正文被丢掉，不报错，引用和检索都少一块。
      - **丢页原因**：差集里每一页再过一遍分类——空文本 / 清洗后为空 / 导航页
        （目录、索引、章扉页）/ 首条目录项之前 / 区间内未分到页 / 这份来源没有目录区间。
        前两类与导航页是**按设计丢的**；「首条目录项之前」与「区间内未分到页」是**真丢正文**
        （页码偏移、目录截断、按目录分页漏 else 分支），计入 `MISSING_CONTENT_MAX` 阈值；
        「没有目录区间」那一类单列待确认，不混进阈值。
        另外把「判成章扉页却留着大段正文」的页记为**疑似误杀**：导航页判定用的
        「点线行 ≥5 且过半」是近似规则，正文页只要列了 5 条以上编号项就会被整页吃掉。
        它是启发式（表格行与短标题不算正文），只进待确认，不做硬断言。
     - **页精度**：块级页码之后，同一节切出的每个子块只声明自己真正覆盖的页，
       于是「part > 0 就是虚报」不再成立；硬断言换成**同一节的相邻 part 至多共享切分处那一页**，
       再加一条**同一页不得同时落在两个目录节里**。改前每个子块复制整节 pages，
       相邻 part 会重叠整节（3.21 倍），这一条照样报错；它同样不需要任何近似判断。
       单页自身超限被按段切开时，同一页出现在相邻两块里是正当的，因此不按次数判错，
       只把「跨块共用的页」当指标报出来（本样本改后应为 0）。
  3. 向量层按 source 整批替换写库，所以块 id 唯一性决定替换后能不能定位到具体块。
  4. 向量层产物 backend/data/lancedb 的 chunks 表：backend/offline/vector/entrance.py 写库成功之后才落 step4，
     所以两边的块 id 与页码必须一模一样。差一个就是「半成品入库」——2026-10-08 真发生过：
     embed 超时失败，step4 已经是新的 138 块，库里还是旧的 137 行、78 块页码不同；
     而在线检索读的是库，于是「分块改好了」这句话在线上并不成立。这一步就是查它。

  签名法（拿源页正文的连续 CJK 串去块正文里找）只作辅助观察：源页是密集 markdown，
  清洗后签名串可能被打断，会漏判，**不能**用来算页覆盖与页精度。

  公开默认样例：backend/data/result/demo_handbook/（自造手册）。本机大手册验收改顶部 SOURCE_PATH / CHUNKS_PATH。

怎么跑（本文件入口）
  职责：读三个产出，打印一组数字与判定，按检查结果决定退出码。
  入口步骤：
    1. load_inputs — 读可分块原文与分块产出。
    2. check_ingest — 按原理 1～3 算页覆盖、丢页原因分类、页精度、id 唯一性、token 分布、tags 唯一数。
    3. load_store_records / check_store_sync — 读 backend/data/lancedb 同来源的块，按原理 4 比 id 与页码。
    4. 打印报告；硬不变量被打破时 exit 1。
  输入：下面常量 SOURCE_PATH / CHUNKS_PATH；库目录取 conf.LANCEDB_DIR。
  前置：无。不调模型；库同步那一步要求 backend/data/lancedb 已入库（读不到即算未过）。
  产出：控制台报告；退出码 0=通过，1=有硬断言未过（STRICT 时任一问题都算未过）。
  对外：check_ingest、classify_missing_pages 与 check_store_sync（纯函数，返回结果 dict，供 pytest 直接调用）。
  跑法：仓库根 uv run python scripts/ingest/verify_ingest.py
"""

from __future__ import annotations

import faulthandler
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT 进 sys.path；库目录取 conf.LANCEDB_DIR

from backend.offline.vector.chunk import (
    NAVIGATION_CHAPTER_COVER,
    clean_page_text,
    estimate_tokens_by_chars,
    is_toc_line,
    navigation_page_reason,
)

# ========== 按需修改 ==========
SOURCE_PATH: Final[str] = "backend/data/result/demo_handbook/chunkable_source.json"
CHUNKS_PATH: Final[str] = "backend/data/result/demo_handbook/step4_chunks.json"
STRICT: Final[bool] = False
RUN_LIMIT_SEC: Final[int] = 300
# 签名串最短长度；太短会在别的页里偶然命中
SIGNATURE_CHARS: Final[int] = 24
# 「有正文、又不是导航页」的丢页上限：这类页没有别的解释，只能是页码偏移、
# 目录截断或按目录分页的算法有洞丢掉的正文，超过这个数就报硬问题。设 0 表示一页都不许丢。
MISSING_CONTENT_MAX: Final[int] = 0
# 判成章扉页的页里，长正文行合计超过这个数 → 记「疑似误杀」（启发式，只进待确认）
SUSPECT_NAVIGATION_BODY_CHARS: Final[int] = 200
# 一行至少这么长才算「正文句」，用来把 markdown 表格行与短标题排除在残留正文之外
SUSPECT_LINE_MIN_CHARS: Final[int] = 40
# ==============================

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{%d,}" % SIGNATURE_CHARS)
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]{%d,}" % SIGNATURE_CHARS)


def strip_spaces(text: str) -> str:
    """去掉全部空白字符，供跨清洗方式的正文比对。"""
    return re.sub(r"\s+", "", text or "")


def page_signature(raw_text: str) -> str:
    """
    取一页正文的签名串：最长连续 CJK 串，没有则退回连续字母数字串。

    功能：给「这页的正文进没进这个块」用的指纹。太短的页没有签名。
    应用场景：check_ingest 的辅助观察，不进硬断言。
    输入：raw_text — 源页原始文本。
    输出：签名串；取不到则空串。
    """
    compact = strip_spaces(raw_text)
    if len(compact) < SIGNATURE_CHARS:
        return ""
    best = ""
    for pattern in (_CJK_RUN, _ALNUM_RUN):
        for match in pattern.finditer(compact):
            if len(match.group(0)) > len(best):
                best = match.group(0)
    return best


def signature_hits(chunk_text: str, signatures: dict) -> set:
    """
    块正文里能找到签名的源页（辅助观察，会漏判）。

    功能：逐页签名在块正文里做子串查找。
    应用场景：页精度的二次确认。
    输入：chunk_text — 已去空白的块正文；signatures — {页号: 签名串}。
    输出：页号集合。
    """
    return {page_no for page_no, signature in signatures.items() if signature and signature in chunk_text}


def token_stats(chunks: list[dict]) -> dict:
    """
    块 token 分布的粗估统计。

    功能：用 estimate_tokens_by_chars 逐块估算，给出 min / 中位 / p90 / max。
    应用场景：分块策略改动的效果对比。
    输入：chunks — 知识块列表。
    输出：{count, min, median, p90, max}。
    """
    values = sorted(estimate_tokens_by_chars(chunk.get("text", "")) for chunk in chunks)
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "max": 0}
    return {
        "count": len(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "p90": values[min(len(values) - 1, int(len(values) * 0.9))],
        "max": values[-1],
    }


# 丢页原因标签（报告里直接打出来）
REASON_EMPTY_RAW = "空文本"
REASON_EMPTY_CLEANED = "清洗后为空"
REASON_FRONT_MATTER = "首条目录项之前"
REASON_RANGE_HOLE = "区间内未分到页"
REASON_NO_CATALOGUE = "来源没有目录区间"


def navigation_leftover_prose(text: str) -> int:
    """
    判成章扉页的页里，去掉目录行与表格行之后剩下的长正文行字符数。

    功能：章扉页本该只剩小节目录；残留正文多，说明是正文页被「点线行 ≥5 且过半」这条规则吃掉了。
          markdown 表格行（以 `|` 起止）与短标题行不算正文——真实手册的章扉页正是这种残留。
    应用场景：classify_missing_pages 的「疑似误杀」启发式，只进待确认，不进硬断言。
    输入：text — 该页原始 markdown。
    输出：长正文行的字符数合计。
    """
    total = 0
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if len(line) < SUSPECT_LINE_MIN_CHARS:
            continue
        if is_toc_line(line) or line.startswith("|") or line.endswith("|"):
            continue
        total += len(line)
    return total


def classify_missing_pages(
    source_obj: dict,
    missing: list,
    *,
    suspect_body_chars: int = SUSPECT_NAVIGATION_BODY_CHARS,
) -> dict:
    """
    给「一个字都没进任何块」的源页逐个贴原因标签。

    功能：按固定次序判一页为什么没进块——源页空 → 导航页（目录 / 索引 / 章扉页）→
          清洗后为空 → 这份来源根本没有目录区间 → 页码在首条目录项之前 → 区间内未分到页。
          只有「首条目录项之前」与「区间内未分到页」是**真丢正文**；
          空页、导航页是按设计丢的，没有目录区间那一类单列。
    应用场景：check_ingest 的丢页告警；命令行报告。
    输入：source_obj — 可分块原文；missing — 没进任何块的页号列表；suspect_body_chars — 疑似误杀阈值。
    输出：{reasons, by_reason, counts, content_pages, front_matter_pages, hole_pages,
           no_catalogue_pages, suspect_navigation_pages, first_catalogue_start}。
    """
    pages = source_obj.get("pages") or []
    page_texts = {
        int(page["page"]): page.get("text", "")
        for page in pages
        if str(page.get("page", "")).isdigit()
    }
    starts = [
        int(catalogue["start_page"])
        for catalogue in (source_obj.get("catalogue_ranges") or [])
        if str(catalogue.get("start_page", "")).isdigit()
    ]
    first_start = min(starts) if starts else None

    reasons = {}
    for page_no in missing:
        raw = page_texts.get(page_no, "")
        if not (raw or "").strip():
            reasons[page_no] = REASON_EMPTY_RAW
            continue
        navigation = navigation_page_reason(raw)
        if navigation:
            reasons[page_no] = navigation
            continue
        if not clean_page_text(raw):
            reasons[page_no] = REASON_EMPTY_CLEANED
            continue
        if first_start is None:
            # 没有目录区间这一支本来就把 pages 写成 []，不是按目录分页漏了
            reasons[page_no] = REASON_NO_CATALOGUE
        elif page_no < first_start:
            reasons[page_no] = REASON_FRONT_MATTER
        else:
            # 目录区间首尾相接，页码 ≥ 首条的页必定落进某一段；落不进就是按目录分页漏了它
            reasons[page_no] = REASON_RANGE_HOLE

    by_reason = {}
    for page_no, reason in reasons.items():
        by_reason.setdefault(reason, []).append(page_no)
    for page_list in by_reason.values():
        page_list.sort()
    content_pages = sorted(
        page_no
        for page_no, reason in reasons.items()
        if reason in (REASON_FRONT_MATTER, REASON_RANGE_HOLE)
    )
    suspects = sorted(
        page_no
        for page_no, reason in reasons.items()
        if reason == NAVIGATION_CHAPTER_COVER
        and navigation_leftover_prose(page_texts.get(page_no, "")) > suspect_body_chars
    )
    return {
        "reasons": reasons,
        "by_reason": by_reason,
        "counts": {reason: len(page_list) for reason, page_list in by_reason.items()},
        "content_pages": content_pages,
        "front_matter_pages": by_reason.get(REASON_FRONT_MATTER, []),
        "hole_pages": by_reason.get(REASON_RANGE_HOLE, []),
        "no_catalogue_pages": by_reason.get(REASON_NO_CATALOGUE, []),
        "suspect_navigation_pages": suspects,
        "first_catalogue_start": first_start,
    }


def check_ingest(
    source_obj: dict,
    chunks: list[dict],
    *,
    missing_content_max: int = MISSING_CONTENT_MAX,
    suspect_body_chars: int = SUSPECT_NAVIGATION_BODY_CHARS,
) -> dict:
    """
    对一份来源的两个产出做全部检查。

    功能：页覆盖、**丢页原因分类**、页精度（相邻 part 至多共享切分页、同一页不得跨节）、
          id 唯一性、token 分布、tags 唯一数。
    应用场景：命令行入口；pytest 直接调用（纯函数，不读盘）。
    输入：source_obj — 可分块原文；chunks — 知识块列表；
          missing_content_max — 「真丢正文」的丢页上限，超过即硬问题；
          suspect_body_chars — 判成章扉页的页残留正文阈值，超过记「疑似误杀」。
    输出：{source, pages_total, pages_declared, pages_missing, missing_pages,
           missing_classification, missing_content_max, ranges, chunks,
           ids_duplicated, declared_page_slots, page_precision, duplicated_pages,
           overlapped_pairs, overlapped_sections, cross_section, continuation_blocks, worst_chunk,
           tag_count, tag_combo_count, tokens, problems, hard_problems}。
    """
    pages = source_obj.get("pages") or []
    ranges = source_obj.get("catalogue_ranges") or []
    tags = source_obj.get("tags") or []

    page_numbers = [int(page["page"]) for page in pages if str(page.get("page", "")).isdigit()]
    page_texts = {int(page["page"]): page.get("text", "") for page in pages}
    signatures = {page_no: page_signature(text) for page_no, text in page_texts.items()}

    # 页覆盖：块声明页的并集（与 eval_kb.py 的页判据同一判据：块自称覆盖哪些页）
    declared_union = set()
    declared_slots = 0
    page_claims = Counter()
    sections = {}
    continuation_blocks = 0
    worst = {"id": "", "declared": 0, "title": "", "part": 0}
    all_text = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_pages = [int(page) for page in metadata.get("pages", [])]
        part = metadata.get("part", 0)
        title = metadata.get("catalogue_title", "")
        declared_union |= set(chunk_pages)
        declared_slots += len(chunk_pages)
        page_claims.update(chunk_pages)
        all_text.append(strip_spaces(chunk.get("text", "")))
        if part and part > 0:
            continuation_blocks += 1
        if len(chunk_pages) > worst["declared"]:
            worst = {
                "id": chunk["id"],
                "title": title,
                "declared": len(chunk_pages),
                "part": part,
            }
        sections.setdefault(title, []).append((part, set(chunk_pages)))

    # 页精度（块级页码）：一节的各 part 是**连续分页区间**，不是整节页的副本。
    # 所以相邻 part 至多共享「切分处那一页」；交集 ≥ 2 页就是页码又退回整节复制（改前 3.21 倍）。
    # 单页自身超限被按段切开时同一页确实会出现在相邻两块里，那是正当的，故不按次数判错。
    overlapped = []
    for title, blocks in sections.items():
        ordered = sorted(blocks, key=lambda item: item[0])
        for (_, left), (_, right) in zip(ordered, ordered[1:]):
            shared = left & right
            if len(shared) > 1:
                overlapped.append((title, sorted(shared)))
    # 同一页不能同时属于两个目录节：那是 assign_content_to_catalogue 归错了
    section_of_page = {}
    cross_section = set()
    for title, blocks in sections.items():
        for _, pages in blocks:
            for page in pages:
                if section_of_page.setdefault(page, title) != title:
                    cross_section.add(page)
    cross_section = sorted(cross_section)
    duplicated_pages = sorted(page for page, count in page_claims.items() if count > 1)
    page_precision = declared_slots / len(declared_union) if declared_union else 1.0

    missing = sorted(set(page_numbers) - declared_union)
    classification = classify_missing_pages(source_obj, missing, suspect_body_chars=suspect_body_chars)

    ids = [chunk["id"] for chunk in chunks]
    duplicated_ids = sorted({chunk_id for chunk_id, count in Counter(ids).items() if count > 1})
    tag_combos = {
        json.dumps(chunk.get("metadata", {}).get("tags", []), ensure_ascii=False, sort_keys=True)
        for chunk in chunks
    }
    # 辅助：整库正文拼起来做一次签名观察，只报数字，不判对错
    merged_text = "\n".join(all_text)
    signed_pages = signature_hits(merged_text, signatures)

    hard_problems = []
    problems = []
    if duplicated_ids:
        hard_problems.append(f"块 id 重复 {len(duplicated_ids)} 个：{duplicated_ids[:5]}")
    if overlapped:
        sample = "、".join(f"{title}（{len(shared)} 页）" for title, shared in overlapped[:3])
        hard_problems.append(
            f"{len(overlapped)} 处相邻 part 的声明页重叠 ≥2 页：{sample}；"
            f"块级页码下相邻两块至多共享切分处那一页，重叠更多说明页码又退回整节复制"
            f"（声明页次 {declared_slots} / 并集 {len(declared_union)} = {page_precision:.2f} 倍）"
        )
    if cross_section:
        hard_problems.append(
            f"{len(cross_section)} 页同时落在两个目录节里（assign_content_to_catalogue 归错）："
            f"{cross_section[:5]}"
        )
    # 丢页：先分类再判。「按设计丢的导航页 / 空页」与「真丢正文」必须分开算，
    # 只报一个「N 页有正文，需逐个确认」等于没有红线——正确丢的章扉页和丢掉的正文长得一样。
    content_pages = classification["content_pages"]
    if len(content_pages) > missing_content_max:
        detail = "；".join(
            f"{reason} {len(classification['by_reason'][reason])} 页 "
            f"{classification['by_reason'][reason][:10]}"
            f"{' …' if len(classification['by_reason'][reason]) > 10 else ''}"
            for reason in (REASON_FRONT_MATTER, REASON_RANGE_HOLE)
            if classification["by_reason"].get(reason)
        )
        hard_problems.append(
            f"{len(content_pages)} 页有正文却没进任何块（上限 {missing_content_max}）：{detail}；"
            f"这些页不属于目录、索引、章扉页，只能是页码偏移或目录截断丢的正文，逐个核页号"
        )
    if classification["suspect_navigation_pages"]:
        problems.append(
            f"{len(classification['suspect_navigation_pages'])} 页被判成章扉页却留着大段正文"
            f"（> {suspect_body_chars} 字）：{classification['suspect_navigation_pages'][:10]}；"
            f"「点线行 ≥5 且过半」这条规则可能吃掉了正文页"
        )
    if classification["no_catalogue_pages"]:
        problems.append(
            f"{len(classification['no_catalogue_pages'])} 页有正文但这份来源没有目录区间："
            f"无目录时 pages 恒为 []，不是按目录分页漏了"
        )
    if len(tag_combos) <= 1 and len(chunks) > 1:
        problems.append(f"tags 组合唯一数 = {len(tag_combos)}：文档级标签复制到每块，块级过滤零区分度")

    return {
        "source": source_obj.get("source", ""),
        "pages_total": len(page_numbers),
        "pages_declared": len(declared_union),
        "pages_missing": len(missing),
        "missing_pages": missing,
        "missing_classification": classification,
        "missing_content_max": missing_content_max,
        "ranges": len(ranges),
        "chunks": len(chunks),
        "declared_page_slots": declared_slots,
        "page_precision": page_precision,
        "duplicated_pages": duplicated_pages,
        "overlapped_pairs": len(overlapped),
        "overlapped_sections": sorted({title for title, _ in overlapped}),
        "cross_section": cross_section,
        "continuation_blocks": continuation_blocks,
        "worst_chunk": worst,
        "ids_duplicated": duplicated_ids,
        "tag_count": len(tags),
        "tag_combo_count": len(tag_combos),
        "signature_pages_found": len(signed_pages),
        "tokens": token_stats(chunks),
        "problems": problems,
        "hard_problems": hard_problems,
    }


def check_store_sync(source: str, chunks: list[dict], records: list[dict]) -> dict:
    """
    核对分块产出与向量库是不是同一批块。

    功能：按 id 比 step4 与库内同来源的块，再比每块的页码。写库成功后 step4 才写进磁盘，两边必须一模一样。
    应用场景：命令行入口的库同步核对；pytest 直接调用（纯函数，不读盘）。
    输入：source — 来源名；chunks — step4 的知识块；records — 向量库记录（含 id / source / pages）。
    输出：{store_rows, step4_rows, ids_only_in_step4, ids_only_in_store, page_mismatches, hard_problems}。
    """
    step4_pages = {
        chunk["id"]: sorted(int(page) for page in chunk.get("metadata", {}).get("pages", []))
        for chunk in chunks
    }
    # 只比这份来源：chunks 表里还躺着别的来源
    store_pages = {
        record["id"]: sorted(int(page) for page in record.get("pages") or [])
        for record in records
        if record.get("source") == source
    }
    only_in_step4 = sorted(set(step4_pages) - set(store_pages))
    only_in_store = sorted(set(store_pages) - set(step4_pages))
    page_mismatches = sorted(
        chunk_id
        for chunk_id in set(step4_pages) & set(store_pages)
        if step4_pages[chunk_id] != store_pages[chunk_id]
    )

    hard_problems = []
    if only_in_step4:
        hard_problems.append(
            f"库里缺 {len(only_in_step4)} 块（step4 有、backend/data/lancedb 没有）：{only_in_step4[:5]}；"
            f"入库没写完，再跑 python scripts/ingest/ingest.py（不必重抽页）"
        )
    if only_in_store:
        hard_problems.append(
            f"库里多 {len(only_in_store)} 块（backend/data/lancedb 有、step4 没有）：{only_in_store[:5]}；"
            f"这份来源没被整批替换过"
        )
    if page_mismatches:
        hard_problems.append(
            f"{len(page_mismatches)} 块的页码两边不同：{page_mismatches[:5]}；"
            f"分块产出改了而向量库还是旧的，在线检索读的是旧页码"
        )

    return {
        "store_rows": len(store_pages),
        "step4_rows": len(step4_pages),
        "ids_only_in_step4": only_in_step4,
        "ids_only_in_store": only_in_store,
        "page_mismatches": page_mismatches,
        "hard_problems": hard_problems,
    }


def print_report(result: dict) -> None:
    """把检查结果打成可读报告。"""
    line = "=" * 68
    print(line)
    print(f"来源              {result['source']}")
    print(f"源页 / 声明覆盖    {result['pages_total']} / {result['pages_declared']}"
          f"   未进任何块 {result['pages_missing']}")
    print(f"目录区间 / 知识块  {result['ranges']} / {result['chunks']}")
    print(f"声明页次 / 并集    {result['declared_page_slots']} / {result['pages_declared']}"
          f"   均摊倍数 {result['page_precision']:.2f}")
    print(f"跨块共用的页       {len(result['duplicated_pages'])} 个（单页超限被按段切开时属正常）"
          f"   相邻 part 重叠>1页 {result['overlapped_pairs']} 处")
    print(f"最宽的一块        {result['worst_chunk']['id']}「{result['worst_chunk']['title']}」"
          f" part={result['worst_chunk']['part']} 声明 {result['worst_chunk']['declared']} 页"
          f"   续块 {result['continuation_blocks']} 个")
    print(f"tags / 组合数      {result['tag_count']} / {result['tag_combo_count']}")
    sync = result.get("store_sync")
    if sync is not None:
        print(f"库内同来源块       {sync['store_rows']} 个（step4 {sync['step4_rows']} 个）"
              f"   仅 step4 有 {len(sync['ids_only_in_step4'])} 个"
              f"   仅库里有 {len(sync['ids_only_in_store'])} 个"
              f"   页码不同 {len(sync['page_mismatches'])} 个")
    tokens = result["tokens"]
    print(f"块 token 估算      min={tokens['min']} 中位={tokens['median']} p90={tokens['p90']} max={tokens['max']}")
    classification = result["missing_classification"]
    counts = classification["counts"]
    breakdown = " / ".join(
        f"{reason} {counts[reason]}" for reason in sorted(counts, key=lambda item: (-counts[item], item))
    )
    print(f"丢页原因分类       {breakdown or '无丢页'}")
    for reason in (REASON_FRONT_MATTER, REASON_RANGE_HOLE, REASON_NO_CATALOGUE):
        page_list = classification["by_reason"].get(reason)
        if page_list:
            print(f"  {reason} {len(page_list)} 页：{page_list[:20]}"
                  f"{' …' if len(page_list) > 20 else ''}")
    if classification["suspect_navigation_pages"]:
        print(f"  疑似误杀 {len(classification['suspect_navigation_pages'])} 页："
              f"{classification['suspect_navigation_pages'][:20]}（判成章扉页却留着大段正文）")
    print(f"真丢正文            {len(classification['content_pages'])} 页"
          f"（上限 {result['missing_content_max']}）：{classification['content_pages'][:20]}"
          f"{' …' if len(classification['content_pages']) > 20 else ''}")
    print(f"签名辅助观察       整库正文里能定位到 {result['signature_pages_found']} / {result['pages_total']} 页"
          f"（密集 markdown 会漏判，仅作参考）")
    print(line)
    for index, problem in enumerate(result["hard_problems"], start=1):
        print(f"硬断言 {index}. {problem}")
    for index, problem in enumerate(result["problems"], start=1):
        print(f"待确认 {index}. {problem}")
    if not result["hard_problems"] and not result["problems"]:
        print("检查结果          全部通过")
    elif not result["hard_problems"]:
        print("检查结果          硬断言通过，有待确认项")


def load_store_records(lancedb_dir: Path) -> list[dict]:
    """
    读向量库 chunks 表的 payload。

    功能：复用 LanceVectorStore.load_payload（只读 payload 列，不读 vector 列、不建 BM25）。
    应用场景：命令行入口的库同步核对。
    输入：lancedb_dir — backend/data/lancedb 目录。
    输出：记录列表，每条含 id / text / metadata / source / pages。
    异常：表缺失或没有行 → FileNotFoundError；embedder 不一致 → ValueError。
    """
    # 重依赖只在这个入口导入：lancedb 一导入就起 asyncio 桥与线程池，纯函数与单测不必付这个代价
    from backend.offline.vector.store import LanceVectorStore

    records, _embedder_name, _dim = LanceVectorStore(lancedb_dir).load_payload()
    return records


def load_inputs(source_path: Path, chunks_path: Path) -> tuple[dict, list]:
    """
    读两个产出文件。

    功能：JSON 解析；文件不存在直接抛。
    应用场景：命令行入口。
    输入：source_path — chunkable_source.json；chunks_path — step4_chunks.json。
    输出：(可分块原文, 知识块列表)。
    异常：文件不存在 → FileNotFoundError；JSON 坏 → json.JSONDecodeError。
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"未找到可分块原文：{source_path}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"未找到分块产出：{chunks_path}")
    source_obj = json.loads(source_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return source_obj, chunks


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. 读可分块原文与分块产出
    source_obj, chunks = load_inputs(Path(SOURCE_PATH), Path(CHUNKS_PATH))
    print(f"[1] 已读 {SOURCE_PATH} / {CHUNKS_PATH}", flush=True)
    # 2. 页覆盖、丢页分类、页精度、id、token、tags
    result = check_ingest(source_obj, chunks)
    print("[2] check_ingest 完成", flush=True)
    # 3. 读 backend/data/lancedb 同来源的块，比 id 与页码
    try:
        records = load_store_records(conf.LANCEDB_DIR)
        store_sync = check_store_sync(source_obj.get("source", ""), chunks, records)
    except (FileNotFoundError, ValueError) as exc:
        store_sync = check_store_sync(source_obj.get("source", ""), chunks, [])
        store_sync["hard_problems"].append(f"向量库读不出来：{exc}")
    result["store_sync"] = store_sync
    result["hard_problems"].extend(store_sync["hard_problems"])
    print("[3] check_store_sync 完成", flush=True)
    # 4. 打印报告；硬不变量被打破时 exit 1
    print_report(result)
    sys.stdout.flush()

    if result["hard_problems"]:
        raise SystemExit(1)
    if STRICT and result["problems"]:
        raise SystemExit(1)
    raise SystemExit(0)
