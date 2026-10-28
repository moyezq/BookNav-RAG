"""向量模型服务。权重在本包同级 models/，不进仓。"""

__all__ = ["MODEL_DIR"]

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
