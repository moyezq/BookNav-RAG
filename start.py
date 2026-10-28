# -*- coding: utf-8 -*-
"""
一键启动：向量服务、问答服务、问答页。

职责：检查环境 → 检查权重与端口 → 起三个子进程 → 等就绪 → 打开浏览器 → Ctrl+C 一起收掉。
不入库、不下载权重；缺权重或尚未入库时打印下一步并退出。

怎么跑：在仓库根执行 `uv run python start.py`（先 `uv sync` 并装好前端 node_modules）。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

EMBED_HOST = "127.0.0.1"
EMBED_PORT = 8030
ASK_HOST = "127.0.0.1"
ASK_PORT = 8031
WEB_PORT = 8032

ASK_URL = f"http://{ASK_HOST}:{ASK_PORT}/"
ASK_DOCS_URL = f"http://{ASK_HOST}:{ASK_PORT}/docs"
WEB_URL = f"http://127.0.0.1:{WEB_PORT}/"
EMBED_URL = f"http://{EMBED_HOST}:{EMBED_PORT}/v1/embeddings"

OPEN_BROWSER = True
# 总等待上限：向量服务加载 bge-m3 在 CPU 上可能要一两分钟
EMBED_READY_TIMEOUT_SEC = 300.0
POLL_INTERVAL_SEC = 0.5

REQUIRED_PACKAGES = (
    "fastapi",
    "uvicorn",
    "lancedb",
    "torch",
    "sentence_transformers",
    "transformers",
    "requests",
    "langgraph",
    "langchain_openai",
)


def step1_check_python() -> bool:
    """打印解释器，并检查本仓 Python 依赖是否在当前环境。"""
    print(f"解释器：{sys.executable}")
    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        print("当前环境缺依赖：" + "、".join(missing))
        print("先执行：uv sync")
        return False
    return True


def step2_check_frontend() -> bool:
    """检查 npm 与 frontend/node_modules。"""
    if shutil.which("npm") is None:
        print("PATH 里找不到 npm：请安装 Node.js ≥ 18。")
        return False
    if not (FRONTEND_DIR / "node_modules").exists():
        print("缺前端依赖：先执行 cd frontend && npm install")
        return False
    return True


def step3_check_weights() -> bool:
    """权重目录里要有 pytorch_model.bin 或 model.safetensors。"""
    from embed import MODEL_DIR

    has_bin = (MODEL_DIR / "pytorch_model.bin").is_file()
    has_safe = (MODEL_DIR / "model.safetensors").is_file()
    if has_bin or has_safe:
        print(f"向量权重：{MODEL_DIR}")
        return True
    print(f"未找到向量权重：{MODEL_DIR}/pytorch_model.bin 或 model.safetensors")
    print("下一步：python embed/download_embed.py")
    return False


def step4_check_knowledge_base() -> bool:
    """问答进程启动时要读非空的 chunks 表；没有就别起，避免立刻崩。"""
    import conf
    from offline.vector.store import LanceVectorStore

    store = LanceVectorStore(conf.LANCEDB_DIR)
    try:
        records, embedder_name, dim = store.load_payload()
    except FileNotFoundError as exc:
        print(f"尚未入库：{exc}")
        print("下一步（样例 PDF 已在 data/source/demo_handbook.pdf）：")
        print("  1. 确认根目录 .env 已填 DEEPSEEK_API_KEY（入库抽标签与问答都要用）")
        print("  2. 另开终端起向量服务：uv run python -m embed.server")
        print("  3. uv run python scripts/ingest/ingest.py")
        print("  4. 停掉临时向量服务后，再跑 uv run python start.py")
        return False
    print(f"知识库：{len(records)} 块，embedder={embedder_name}，dim={dim}")
    return True


def step5_check_ports() -> bool:
    """端口被占用就别再起第二个。"""
    required = [EMBED_PORT, ASK_PORT, WEB_PORT]
    busy = [str(port) for port in required if _port_in_use(port)]
    if busy:
        print("这些端口已被占用：" + "、".join(busy))
        print(f"可能服务已经在跑：{ASK_DOCS_URL} 、{WEB_URL}")
        print("要重启就先停掉旧进程，再运行本脚本。")
        return False
    return True


def step6_start_embed() -> subprocess.Popen:
    """起向量服务（加载模型后才监听端口）。"""
    print(f"启动向量服务 → {EMBED_URL}")
    return subprocess.Popen([sys.executable, "-m", "embed.server"], cwd=str(PROJECT_DIR))


def step7_start_ask() -> subprocess.Popen:
    """起问答服务。"""
    print(f"启动问答服务 → {ASK_DOCS_URL}")
    return subprocess.Popen([sys.executable, "-m", "online.server"], cwd=str(PROJECT_DIR))


def step8_start_web() -> subprocess.Popen:
    """起问答页。"""
    print(f"启动问答页 → {WEB_URL}")
    npm = shutil.which("npm")
    assert npm is not None
    return subprocess.Popen([npm, "run", "dev"], cwd=str(FRONTEND_DIR))


def step9_wait_ready(
    embed: subprocess.Popen,
    ask: subprocess.Popen,
    web: subprocess.Popen,
) -> bool:
    """
    并行等三个服务就绪。

    不要串行：向量加载要一两分钟，若等它结束再去探问答页，会把「其实早已起来的 Vite」
    误判成超时，然后 finally 把三个进程一起杀掉（表现为页面突然断开）。
    """
    print("等待就绪（向量加载可能较久；问答页起来后也会先打一行）…")
    deadline = time.monotonic() + EMBED_READY_TIMEOUT_SEC
    embed_ready = ask_ready = web_ready = False
    while time.monotonic() < deadline:
        if embed.poll() is not None:
            print(f"向量服务进程已退出（退出码 {embed.returncode}），错误见它上面的日志。")
            return False
        if ask.poll() is not None:
            print(f"问答服务进程已退出（退出码 {ask.returncode}），错误见它上面的日志。")
            return False
        if web.poll() is not None:
            print(f"问答页进程已退出（退出码 {web.returncode}），错误见它上面的日志。")
            return False

        if not embed_ready and _port_in_use(EMBED_PORT) and _probe_embed():
            embed_ready = True
            print(f"向量服务就绪：{EMBED_URL}")
        if not ask_ready and _http_ok(ASK_URL):
            ask_ready = True
            print(f"问答服务就绪：{ASK_DOCS_URL}")
        # 端口已听也算：避免只绑 localhost/IPv6 时 127.0.0.1 的 HTTP 探测假失败
        if not web_ready and (_http_ok(WEB_URL) or _port_in_use(WEB_PORT)):
            web_ready = True
            print(f"问答页就绪：{WEB_URL}")

        if embed_ready and ask_ready and web_ready:
            return True
        time.sleep(POLL_INTERVAL_SEC)

    missing = []
    if not embed_ready:
        missing.append(f"向量服务（{EMBED_URL}）")
    if not ask_ready:
        missing.append(f"问答服务（{ASK_URL}）")
    if not web_ready:
        missing.append(f"问答页（{WEB_URL}）")
    print(f"等待就绪超时（{EMBED_READY_TIMEOUT_SEC:.0f} 秒），未就绪：{'、'.join(missing)}")
    return False


def step10_open_browser() -> None:
    """打开浏览器（可用顶部 OPEN_BROWSER 关掉）。"""
    if OPEN_BROWSER:
        webbrowser.open(WEB_URL)


def run() -> int:
    """只做编排：逐步检查与启动，Ctrl+C 时把子进程一起收掉。"""
    checks = (
        step1_check_python,
        step2_check_frontend,
        step3_check_weights,
        step4_check_knowledge_base,
        step5_check_ports,
    )
    for check in checks:
        if not check():
            return 1

    embed = ask = web = None
    try:
        embed = step6_start_embed()
        ask = step7_start_ask()
        web = step8_start_web()
        if not step9_wait_ready(embed, ask, web):
            return 1
        step10_open_browser()
        print("服务在跑，按 Ctrl+C 一起停止。")
        while True:
            if embed.poll() is not None or ask.poll() is not None or web.poll() is not None:
                print("有服务退出了，一并收尾。")
                return 1
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止…")
        return 0
    finally:
        _stop(web)
        _stop(ask)
        _stop(embed)


def _stop(process: subprocess.Popen | None) -> None:
    """结束子进程树。Windows 上 npm 会再拉 node，只杀父进程会占端口。"""
    if process is None or process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _port_in_use(port: int) -> bool:
    """TCP 连得上就说明有人在监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _http_ok(url: str) -> bool:
    """能取到 HTTP 响应（哪怕是 404）就算服务已经起来了。"""
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _probe_embed() -> bool:
    """POST 一条极短文本，确认模型已加载且接口可用。"""
    body = json.dumps({"input": "ping", "model": "bge-m3"}).encode("utf-8")
    request = urllib.request.Request(
        EMBED_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(run())
