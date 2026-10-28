# -*- coding: utf-8 -*-
"""
基础信息：文件进来先认路径、stem、大小、扩展名。

使用前注意事项：不调 anyurl2text，不读文件正文。
"""

from __future__ import annotations

from pathlib import Path


def probe_file(path: Path) -> dict:
    """
    抽出一份文件的基础信息。

    功能：stat 大小；扩展名去掉点并小写。
    应用场景：抽取层第一步，给路由用。
    输入：path — 源文件。
    输出：{path, source, name, size_bytes, ext}。
    异常：文件不存在 → FileNotFoundError。
    """
    if not path.is_file():
        raise FileNotFoundError(f"源文件不存在: {path}")
    return {
        "path": str(path.resolve()),
        "source": path.stem,
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "ext": path.suffix.lstrip(".").lower(),
    }
