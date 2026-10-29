# -*- coding: utf-8 -*-
"""
本地 bge-m3 → OpenAI 兼容 POST /v1/embeddings。

职责：向量模型服务实现与进程入口。
前置：service/embed/models/ 下有 pytorch_model.bin 或 model.safetensors；可用 python service/embed/download_embed.py 下载。
对外：serve(host, port)；POST /v1/embeddings。
怎么跑：在仓库根 `python -m service.embed.server`。不要 `python service/embed/server.py`。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from service.embed import MODEL_DIR

_WEIGHT_FILE_NAMES = ("pytorch_model.bin", "model.safetensors")

# ========== 按需修改 ==========
HOST = "127.0.0.1"
PORT = 8030
BATCH_SIZE = 8
EMBED_MODEL_NAME = "bge-m3"
# ==============================

_model = None
_encode_lock = threading.Lock()


def _weight_files() -> list[Path]:
    """仓内已有的权重文件；两种发布形态都认。"""
    return [MODEL_DIR / name for name in _WEIGHT_FILE_NAMES if (MODEL_DIR / name).is_file()]


def load_model() -> SentenceTransformer:
    """
    从 service/embed/models/ 加载。

    功能：有 CUDA 用 GPU，否则 CPU。
    应用场景：服务启动一次。
    输入：无，读 MODEL_DIR。
    输出：SentenceTransformer。
    异常：目录或权重缺失 → FileNotFoundError。
    """
    if not _weight_files():
        raise FileNotFoundError(
            f"未找到权重 {MODEL_DIR}/pytorch_model.bin 或 model.safetensors，"
            "先跑 python service/embed/download_embed.py"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 {MODEL_DIR} device={device}")
    return SentenceTransformer(str(MODEL_DIR), device=device)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    把一批文本编成 L2 归一向量。

    功能：内部按 BATCH_SIZE 分批，加锁避免并发抢模型。
    应用场景：/v1/embeddings。
    输入：texts — 字符串列表。
    输出：与 texts 等长的向量列表。
    """
    if not texts:
        return []
    with _encode_lock:
        vectors = _model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return [row.tolist() for row in vectors]


def parse_input(payload: dict) -> list[str]:
    """OpenAI 的 input：字符串或字符串列表。"""
    raw = payload["input"]
    if isinstance(raw, str):
        return [raw]
    return list(raw)


class EmbeddingHandler(BaseHTTPRequestHandler):
    """只处理 POST /v1/embeddings。"""

    def log_message(self, format, *args) -> None:
        """启动后只打本请求的路径与状态。"""
        print(self.address_string(), "-", format % args)

    def _write_json(self, status: int, body: dict) -> None:
        """写 JSON 响应。"""
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        """
        接收 OpenAI 兼容 embedding 请求。

        功能：读 input，返回 data[].embedding。
        应用场景：向量层入库、在线召回。
        """
        if self.path != "/v1/embeddings":
            self._write_json(404, {"error": {"message": f"未知路径 {self.path}", "type": "invalid_request_error"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        texts = parse_input(payload)
        print(f"embed n={len(texts)}")
        vectors = embed_texts(texts)
        rows = [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ]
        self._write_json(
            200,
            {
                "object": "list",
                "data": rows,
                "model": EMBED_MODEL_NAME,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )


def serve(host: str, port: int) -> None:
    """
    加载模型并阻塞监听。

    功能：ThreadingHTTPServer 提供 /v1/embeddings。
    应用场景：python -m service.embed.server。
    输入：host / port — 绑定地址。
    输出：无，阻塞直到进程退出。
    """
    global _model
    _model = load_model()
    server = ThreadingHTTPServer((host, port), EmbeddingHandler)
    print(f"embeddings http://{host}:{port}/v1/embeddings")
    server.serve_forever()


if __name__ == "__main__":
    serve(HOST, PORT)
