# -*- coding: utf-8 -*-
"""
Kev-0.8B 权重下载（ModelScope），并把底座摆成 kev 认的 HF 缓存布局。

职责：只准备权重。不起服务、不加载模型；用完 torch 只做一次 head.pt 的只读解析。
       为什么还要摆 HF 缓存：kev 加载 adapter 时用 `transformers` 按 head.pt 里的 `base`+`base_revision`
       去 HF 缓存找底座（repo id，不是路径），所以底座下到哪儿都必须摆成 cache 的形状。
前置：modelscope + torch 可 import（本仓 uv 环境里都有）。
输出：models/kev-0.8b/（adapter + head.pt + tokenizer）、models/base/（Qwen3.5-0.8B-Base 原文）、
       models/hf/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/<sha>/（底座的缓存视图，硬链接，不占第二份空间）。
怎么跑：uv run python scripts/kev08/download_kev08.py
换尺寸或换镜像改下面的常量，不加命令行参数。
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import torch
from modelscope import snapshot_download

# ========== 按需修改 ==========
MODEL_ID = "jaredpalmer/kev-0.8b"
MODEL_REVISION = "master"  # ModelScope 用分支名，没有 HF 的 v1.0 标签；版本靠下面的 sha256 钉住
BASE_ID = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "master"
MODELS_DIR = Path(__file__).resolve().parent / "models"
# ==============================

CHECKPOINT_PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]
BASE_PATTERNS = ["*.json", "*.safetensors", "*.txt"]
CHECKPOINT_DIR = MODELS_DIR / "kev-0.8b"
BASE_DIR = MODELS_DIR / "base"
HF_HUB_DIR = MODELS_DIR / "hf" / "hub"

# 钉住上游 Kev 1.0 的两份权重（ModelScope API 报的 sha256），换版本时一起改
PINNED_SHA256 = {
    "adapter_model.safetensors": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
    "head.pt": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
}


def _sha256(path: Path) -> str:
    """分块算 sha256，避免把 43 MB 整个读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint() -> Path:
    """
    从 ModelScope 拉 Kev 的 adapter、head.pt 与 tokenizer。

    功能：snapshot_download 到本地目录，再按 sha256 核对是不是钉住的那一版。
    应用场景：只想按本地目录起服务，不让 kev.serve 去联网取 adapter。
    输入：无，读上面的常量。
    输出：checkpoint 目录。
    异常：网络/revision 问题 → modelscope 的异常原样抛出；sha256 不符 → ValueError。
    """
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(CHECKPOINT_DIR),
        allow_file_pattern=CHECKPOINT_PATTERNS,
    )
    for name, expected in PINNED_SHA256.items():
        actual = _sha256(CHECKPOINT_DIR / name)
        if actual != expected:
            raise ValueError(f"{name} sha256 不符：{actual}（期望 {expected}）；上游改过版本就把 PINNED_SHA256 一起改")
    print(f"[权重 1/2] {MODEL_ID}@{MODEL_REVISION} → {CHECKPOINT_DIR}（sha256 已核对）")
    return CHECKPOINT_DIR


def download_base() -> Path:
    """
    从 ModelScope 拉 Qwen3.5-0.8B-Base 的权重与 tokenizer。

    功能：下到 models/base/，保持 ModelScope 的原始文件名。
    应用场景：kev 加载 adapter 时要这个底座；这一步之后不再需要联网。
    输入：无，读上面的常量。
    输出：底座目录。
    异常：网络/revision 问题 → modelscope 的异常原样抛出。
    """
    snapshot_download(
        BASE_ID,
        revision=BASE_REVISION,
        local_dir=str(BASE_DIR),
        allow_file_pattern=BASE_PATTERNS,
    )
    print(f"[权重 2/2] {BASE_ID}@{BASE_REVISION} → {BASE_DIR}")
    return BASE_DIR


def _head_meta(checkpoint_dir: Path) -> dict:
    """读 head.pt 的元数据字典；只读，不写回。"""
    return torch.load(checkpoint_dir / "head.pt", map_location="cpu", weights_only=True)


def build_hf_cache(checkpoint_dir: Path, base_dir: Path) -> Path:
    """
    把底座摆成 HF 缓存里的一份快照，kev 不联网就能按 repo id 找到它。

    功能：读 head.pt 的 base/base_revision，在 HF_HOME/hub 下建
          models--<ns>--<name>/snapshots/<sha>/，逐个文件硬链接（失败退回复制），并写 refs/main。
    应用场景：底座是 ModelScope 下的，但 transformers 只认缓存布局；这样不用改 head.pt、不改 kev。
    输入：checkpoint_dir — adapter 目录；base_dir — 底座原文目录。
    输出：快照目录路径。
    异常：head.pt 里的 base 与 BASE_ID 不一致 → ValueError（说明权重换了底座，本脚本的常量要跟着改）。
    """
    meta = _head_meta(checkpoint_dir)
    base_repo = str(meta.get("base", ""))
    if base_repo != BASE_ID:
        raise ValueError(f"head.pt 说底座是 {base_repo!r}，脚本常量是 {BASE_ID!r}；改常量后重跑")
    sha = str(meta.get("base_revision") or "main")
    repo_dir = HF_HUB_DIR / ("models--" + base_repo.replace("/", "--"))
    snapshot_dir = repo_dir / "snapshots" / sha
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source in base_dir.iterdir():
        if not source.is_file():
            continue
        target = snapshot_dir / source.name
        if target.exists():
            continue
        try:
            os.link(source, target)  # 硬链接：同一份 1.75 GB 不占两处
        except OSError:
            shutil.copy2(source, target)
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(sha, encoding="utf-8")
    print(f"[缓存] {base_repo}@{sha} → {snapshot_dir}")
    return snapshot_dir


def main() -> None:
    """按 adapter → 底座 → HF 缓存的顺序准备，最后打印根目录。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = download_checkpoint()
    base_dir = download_base()
    build_hf_cache(checkpoint_dir, base_dir)
    total_mb = sum(item.stat().st_size for item in MODELS_DIR.rglob("*") if item.is_file()) / 1024 / 1024
    print(f"[权重] 根目录 {MODELS_DIR}（去重前合计 {total_mb:.1f} MB；底座与缓存视图是硬链接）")


if __name__ == "__main__":
    main()
