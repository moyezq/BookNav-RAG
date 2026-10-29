"""
生成可再分发的演示手册 PDF（自造内容，非第三方版权）。
写出 backend/data/source/demo_handbook.pdf 后退出。不入库、不起服务。

怎么跑（本文件入口）
  职责：用 pymupdf 画目录页 + 正文页，写入 SOURCE_DIR / demo_handbook.pdf。
  入口步骤：
    1. build_pdf — TOC_ENTRIES + BODY_PAGES → backend/data/source/demo_handbook.pdf。
  输入：下面常量 TOC_ENTRIES / BODY_PAGES。
  前置：本仓环境能 import pymupdf（pymupdf4llm 的依赖）。
  产出：backend/data/source/demo_handbook.pdf（6 页：目录 + 5 节正文）。
  跑法：仓库根 uv run python scripts/ingest/make_demo_pdf.py
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import backend.conf as conf  # ROOT 在 conf.base；读 yaml 与 .env
from backend.conf import SOURCE_DIR

# ========== 按需修改 ==========
OUT_PATH: Final[Path] = SOURCE_DIR / "demo_handbook.pdf"
RUN_LIMIT_SEC: Final[int] = 300
# ==============================

# 目录印刷页码 = PDF 阅读器页码（与 PdfPageExtract 约定一致）
TOC_ENTRIES: Final[tuple[tuple[str, int], ...]] = (
    ("安全须知", 2),
    ("折叠支架", 3),
    ("展开支架", 4),
    ("充电说明", 5),
    ("规格参数", 6),
)

BODY_PAGES: Final[dict[int, tuple[str, list[str]]]] = {
    2: (
        "安全须知",
        [
            "使用本演示设备前，请阅读本章全部内容。",
            "请勿在潮湿环境中给设备充电；发现外壳破损请停止使用。",
            "儿童请在监护人指导下操作折叠支架，避免夹伤手指。",
        ],
    ),
    3: (
        "折叠支架",
        [
            "折叠支架位于设备背面中部，用于收纳与携带。",
            "按下折叠开关后，左右支架臂会同时向内侧收拢；听到咔嗒声表示已锁止。",
            "折叠过程中请勿用力扳动支架臂，以免损坏铰链。",
            "完全折叠后可将设备放入附带收纳袋。",
        ],
    ),
    4: (
        "展开支架",
        [
            "再次按下折叠开关，左右支架臂会同时向外展开并锁定在支撑角度。",
            "展开后轻推支架臂，确认已锁止再放手；未锁止时设备可能倾倒。",
            "若支架无法展开，先检查折叠开关是否被异物卡住，再重试一次。",
        ],
    ),
    5: (
        "充电说明",
        [
            "本设备使用 5V/2A 适配器，接口位于机身右侧。",
            "充电时指示灯为橙色；充满后变为绿色。",
            "请选择干燥、通风的环境充电，远离火源与易燃物。",
            "连续充电超过十二小时仍未转绿，请停止充电并检查适配器。",
        ],
    ),
    6: (
        "规格参数",
        [
            "产品名称：演示折叠支架设备（虚构型号 DEMO-01）。",
            "外形尺寸：长 180 mm、宽 90 mm、高 12 mm（支架折叠状态）。",
            "质量：约 220 g。",
            "工作温度：0℃～40℃。",
            "本手册为开源演示样例，内容可自由复制与修改。",
        ],
    ),
}


def _wrap_line(text: str, *, width: int) -> list[str]:
    """按字符数粗分行，避免单行超出页面。"""
    if len(text) <= width:
        return [text]
    return [text[i : i + width] for i in range(0, len(text), width)]


def build_pdf(path: Path) -> None:
    """
    用 pymupdf 画目录页 + 正文页，写入 path。

    功能：一页「目 录」+ BODY_PAGES 各一页；CJK 用 china-s，避免嵌字体把体积打到十几 MB。
    应用场景：演示手册缺失或要改文案时重生成。
    输入：path — 目标 PDF。
    输出：无，写入 path。
    """
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    try:
        toc_page = doc.new_page(width=595, height=842)
        # 每条目录用独立 text writer、行距 ≥36pt：过密时 pymupdf4llm 会把多行收成一行空格分隔。
        y = 72.0
        toc_page.insert_text((72, y), "目 录", fontsize=16, fontname="china-s")
        y += 48.0
        for title, page_no in TOC_ENTRIES:
            toc_page.insert_text(
                (72, y),
                f"{title}........{page_no}",
                fontsize=12,
                fontname="china-s",
            )
            y += 36.0

        for page_no in sorted(BODY_PAGES):
            title, paragraphs = BODY_PAGES[page_no]
            page = doc.new_page(width=595, height=842)
            y = 72.0
            page.insert_text((72, y), title, fontsize=14, fontname="china-s")
            y += 36.0
            for paragraph in paragraphs:
                for line in _wrap_line(paragraph, width=28):
                    page.insert_text((72, y), line, fontsize=12, fontname="china-s")
                    y += 22.0
                y += 10.0

        doc.save(path)
    finally:
        doc.close()


if __name__ == "__main__":
    faulthandler.dump_traceback_later(RUN_LIMIT_SEC, exit=True)
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. TOC_ENTRIES + BODY_PAGES → backend/data/source/demo_handbook.pdf
    build_pdf(OUT_PATH)
    print(f"[1] 已写入 {OUT_PATH.relative_to(conf.ROOT)}（{len(BODY_PAGES) + 1} 页）", flush=True)
    print("下一步：python scripts/ingest/extract_pdf.py ，再 python scripts/ingest/ingest.py", flush=True)
