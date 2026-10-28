# -*- coding: utf-8 -*-
"""
手册内嵌图的 VLM OCR。本阶段流程不调用。

使用前注意事项：
- 接口形状按 OpenAI chat/completions，图用 data URL。
- 命中 cache_dir/<图片名>.txt 则不再打模型。
- 读不出字时模型应回 [NO_TEXT]，调用方不替换该图。
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

RESIZE_MAX_SIDE = 2000

OCR_SYSTEM_PROMPT = """<角色>
Assistant是一个专业的视觉语言模型（VLM），擅长从用户提供的图像中准确地提取文字内容，并以文本形式输出。
</角色>

<任务>
仔细分析用户提供的图像（包括截图、表格、列表、小号字体、界面文字等）。
执行OCR任务，从图像中清晰且准确地提取所有可识别的文字内容；表格请按行列顺序输出，列表按从上到下、从左到右顺序输出。
将识别后的文字严格按图像原始顺序完整呈现。若部分区域不清晰，请输出已能识别的部分，不要因局部模糊而整体不输出。
如果图像模糊什么都读不出，快速返回<ocr_result>[NO_TEXT]</ocr_result>，不要思考。
</任务>

<限制>
Assistant仅提供图像中实际存在的文字，不进行总结、解释或添加任何额外信息。忽略图片中非文字内容，不纠正或猜测不清晰的文字。
</限制>

<输出格式>
将识别后的文字内容严格放到<ocr_result></ocr_result>之间。
如果图像中没有任何文字内容，请输出<ocr_result>[NO_TEXT]</ocr_result>
</输出格式>"""


def _pil_to_jpeg_data_url(image_path: Path) -> str:
    """读图、必要时缩小长边、编成 JPEG data URL。"""
    with Image.open(image_path) as img:
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        width, height = img.size
        if max(width, height) > RESIZE_MAX_SIDE:
            scale = RESIZE_MAX_SIDE / float(max(width, height))
            img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_ocr_result(content: str) -> str:
    """从模型回复里抽出 <ocr_result>；去掉思维链包裹。"""
    text = content or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if "<answer>" in text:
        text = text.split("<answer>", 1)[1].replace("</answer>", "").strip()
    match = re.search(r"<ocr_result>(.*?)</ocr_result>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def ocr_image_file(
    image_path: Path,
    cache_dir: Path,
    api_url: str,
    model_name: str,
    timeout: int = 120,
) -> str:
    """
    对单张图做 OCR，结果写入缓存文件。

    功能：缓存命中直接返回；否则 POST 视觉 chat 接口。
    应用场景：replace_md_images_with_ocr 逐张图。
    输入：image_path — 图片；cache_dir — 缓存目录；api_url / model_name — 视觉接口。
    输出：识别文本或 [NO_TEXT]。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{image_path.name}.txt"
    if cache_file.is_file():
        cached = cache_file.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    data_url = _pil_to_jpeg_data_url(image_path)
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请 OCR 图中全部文字。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    response = requests.post(api_url, json=body, timeout=timeout)
    response.raise_for_status()
    ocr_text = _parse_ocr_result(response.json()["choices"][0]["message"]["content"])
    cache_file.write_text(ocr_text, encoding="utf-8")
    print(f"[vlm-ocr] {image_path.name} len={len(ocr_text)}")
    return ocr_text


def replace_md_images_with_ocr(
    text: str,
    images_dir: Path,
    cache_dir: Path,
    api_url: str,
    model_name: str,
) -> str:
    """
    把 markdown 图片换成 OCR 文本。

    功能：按出现顺序替换；文件找不到或 [NO_TEXT] 则跳过该图。
    应用场景：目录页 / 正文页在按目录分页前可选增强。
    输入：text — 页 markdown；images_dir — pymupdf4llm 出图目录。
    输出：替换后的纯文本串（空白已压平）。
    """
    cleaned = (text or "").replace("\\n", " ").replace("\n", " ").replace("\\r", " ")
    md_images = re.findall(r"!\[[^\]]*\]\([^)]*\)", cleaned)
    for md_image in md_images:
        match = re.match(r"!\[[^\]]*\]\((.*?)\)", md_image)
        if not match:
            continue
        raw_path = match.group(1)
        image_path = Path(raw_path)
        if not image_path.is_file():
            image_path = images_dir / Path(raw_path).name
        if not image_path.is_file():
            continue
        ocr_text = ocr_image_file(image_path, cache_dir, api_url, model_name)
        if ocr_text == "[NO_TEXT]":
            continue
        cleaned = cleaned.replace(md_image, ocr_text, 1)
    return re.sub(r"\s+", " ", cleaned).strip()
