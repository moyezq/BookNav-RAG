# -*- coding: utf-8 -*-
"""
pytest 引导：把仓库根放进导入路径，让测试能 import backend.conf as conf 与各层模块；并提供可写的临时目录夹具。

职责：只在收集测试前改 sys.path；不 mock 业务、不改环境变量（.env 读取仍走 conf 自己的逻辑）。
      另提供 work_dir —— 沙箱里 pytest 自带的 tmp_path 用不了，原因见该夹具的 docstring。
用法：在仓库根跑 python -m pytest。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def work_dir():
    """
    一个可写的临时目录，用例结束整棵树删掉；要写文件的用例用它，不要用 tmp_path。

    功能：在系统临时目录下用**不带 mode 的 mkdir** 建目录。
    应用场景：需要写文件的用例，例如向量层的写文件顺序。
    输入：无。
    输出：Path，本用例独占。
    为什么不用 tmp_path（2026-10-08 实测）：DSH 沙箱里 os.mkdir(path, 0o700) 建出来的目录
      连创建者自己都列不出、写不进（WinError 5 / EACCES），而 pytest 的 basetemp 正是按
      0o700 建的 → 所有 tmp_path 用例在收集阶段就 PermissionError。普通 mkdir 没有这个问题。
    """
    path = Path(tempfile.gettempdir()) / f"book-nav-rag-test-{uuid4().hex[:8]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
