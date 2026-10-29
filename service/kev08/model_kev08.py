# -*- coding: utf-8 -*-
"""
Kev-0.8B 的推理本体：本目录自己实现，不 import kev。

职责：把「一段 state + 若干类型化问题」变成「每个选项一个概率」。全部步骤都在这一个文件里：
      1. 编码：state 用 <|fim_prefix|> 起头；每个问题一行 [<|fim_middle|>] 指令 [<|box_start|>] 选项 [<|box_end|>]… <|fim_suffix|>
      2. 前向：Qwen3.5 的 hybrid 层（Gated DeltaNet）只能走 row 形式——每个问题一行，行首带整段 state
      3. 打分：指针头把「每个选项收尾 token 的隐状态」对「该行最后一个 token（<|fim_suffix|>）的隐状态」做点积
      4. 校准：点积除以 head.pt 里存的温度（0.8B 是 2.35），再 softmax
前置：models/kev-0.8b 下有 head.pt / adapter_config.json / adapter_model.safetensors / tokenizer*；
      models/base 下是 Qwen3.5-0.8B-Base 的原始权重。
与 kev.serve 的唯一差别：它用 state 前缀缓存 + 批量，这里每个问题重算一遍 state（row 形式），两者按 fp32 舍入相等，慢但不是另一种算法。
对外：load_model / answer。
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["MAX_STATE", "Model", "load_model"]

MAX_STATE = 65536  # 与 kev 的 SERVE_MAX_STATE 一致：state token 数上限（含 <|fim_prefix|>）
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class PointerHead(nn.Module):
    """把「选项收尾 token」与「问题收尾 token」投影到同一空间后点积，得到每个选项的分数。"""

    def __init__(self, hidden_size: int, pointer_dim: int = 256) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_size, pointer_dim)
        self.k = nn.Linear(hidden_size, pointer_dim)
        self.scale = 1 / math.sqrt(pointer_dim)
        self.temperature = 1.0

    def forward(self, h_decide: torch.Tensor, h_options: torch.Tensor) -> torch.Tensor:
        """[d], [K,d] → 分数 [K]；eval 模式下除以温度，argmax 不受影响。"""
        z = (self.k(h_options) @ self.q(h_decide)) * self.scale
        return z if self.temperature == 1.0 else z / self.temperature


def user_tokens(tok, text: str) -> list[int]:
    """把调用方文本里的 `<|xxx|>` 改写成 `<¦xxx¦>` 再分词：让调用方无法伪造分隔符。"""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def render(value) -> str:
    """复刻 kev/api.py 的 render()：str 原样，dict 变 `key: value`，list 变 `- item`。"""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"- {render(item).lstrip()}" for item in value)
    return "\n".join(f"{k}: {render(v)}" if not isinstance(v, (dict, list)) else f"{k}:\n{render(v)}" for k, v in value.items())


def option_text(name: str, desc) -> str:
    """选项串：`名字` 或 `名字: 描述`。这就是模型真正看到的那段文本。"""
    return name if desc in (None, "") else f"{name}: {render(desc)}"


class Model:
    """加载好的一个 checkpoint：分词器、主干、指针头、设备与精度。"""

    def __init__(self, tok, backbone, head: PointerHead, device: str, dtype: torch.dtype, run_dir: Path) -> None:
        self.tok = tok
        self.backbone = backbone
        self.head = head
        self.device = device
        self.dtype = dtype
        self.run_dir = run_dir

    def encode(self, state: str, questions: dict) -> tuple[list[int], list[dict]]:
        """
        把一次请求编码成 state 与每个问题的一行。

        功能：state 前缀 + 每个问题的完整 row（row 形式，state 每行重复一遍）。
        应用场景：一次 /v1/systemone 请求。
        输入：state — 原文；questions — {问题名: {type, instructions, criteria?}}。
        输出：(state token id, [{"name", "ids", "decide", "options", "keys", "type"}])。
        异常：state 超过 MAX_STATE → ValueError（调用方转 422）。
        """
        ids = {name: self.tok.convert_tokens_to_ids(token) for name, token in zip("qocd", SPECIAL[1:])}
        state_tokens = user_tokens(self.tok, state)
        state_ids = [self.tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens
        if len(state_ids) > MAX_STATE:
            raise ValueError(f"state 是 {len(state_ids):,} 个 token，超过 {MAX_STATE:,} 上限")
        rows = []
        for name, question in questions.items():
            qtype = question["type"]
            if qtype == "noul":
                criteria = question.get("criteria") or {}
                texts = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
                keys = ["false", "true"]
            elif qtype == "choice":
                texts = [option_text(option, desc) for option, desc in question["criteria"].items()]
                keys = list(question["criteria"])
            else:
                raise ValueError(f"本实现只处理 noul / choice，收到 {qtype!r}")
            branch = [ids["q"]] + user_tokens(self.tok, render(question.get("instructions")))
            ends = []
            for text in texts:
                span = [ids["o"]] + user_tokens(self.tok, text) + [ids["c"]]
                branch += span
                ends.append(len(branch) - 1)
            branch.append(ids["d"])
            row = state_ids + branch
            rows.append(
                {
                    "name": name,
                    "ids": row,
                    "decide": len(row) - 1,
                    "options": [len(state_ids) + end for end in ends],
                    "keys": keys,
                    "type": qtype,
                }
            )
        return state_ids, rows

    @torch.no_grad()
    def probabilities(self, state: str, questions: dict) -> tuple[list[torch.Tensor], int, list[dict]]:
        """
        一次请求里每个问题的选项概率。

        功能：逐行前向，取 `<|fim_suffix|>` 与各 `</opt>` 位置的隐状态，过指针头再 softmax。
        应用场景：/v1/systemone。
        输入：state / questions — 同 encode。
        输出：(每个问题的概率张量, 本次输入 token 数, encode 的行——含 name / keys / type)。
        异常：编码失败（超长 / 未知类型）→ ValueError。
        """
        state_ids, rows = self.encode(state, questions)
        results = []
        total_tokens = len(state_ids)
        for row in rows:
            input_ids = torch.tensor([row["ids"]], device=self.device)
            position_ids = torch.arange(len(row["ids"]), device=self.device).unsqueeze(0)
            hidden = self.backbone(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            ).last_hidden_state[0].float()  # fp32 读数，与 kev 一致
            h_decide = hidden[row["decide"]]
            h_options = hidden[torch.tensor(row["options"], device=self.device)]
            results.append(F.softmax(self.head(h_decide, h_options), -1).cpu())
            total_tokens += len(row["ids"]) - len(state_ids)
        return results, total_tokens, rows


def load_model(run_dir: Path, base_dir: Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> Model:
    """
    加载底座 + LoRA adapter + 指针头。

    功能：AutoModelForCausalLM 取主干（不带 LM head）、peft 合入 adapter、head.pt 恢复指针头与温度。
    应用场景：服务启动一次。
    输入：run_dir — models/kev-0.8b；base_dir — models/base；device / dtype — 运行位置与精度。
    输出：Model。
    异常：文件缺失 → FileNotFoundError；adapter 与底座版本不匹配 → 由 transformers / peft 抛出。
    """
    for name in ("head.pt", "adapter_config.json", "adapter_model.safetensors"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir / name} 不存在；先在 service/kev08 里跑 uv run python download_kev08.py")
    meta = torch.load(run_dir / "head.pt", map_location="cpu", weights_only=True)
    attn = "sdpa" if device.startswith("cuda") else "eager"
    tok = AutoTokenizer.from_pretrained(str(run_dir))
    backbone = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=dtype, attn_implementation=attn).model
    backbone = PeftModel.from_pretrained(backbone, str(run_dir), torch_device=device)
    backbone = backbone.merge_and_unload()  # adapter 合进底座，之后就没有 LoRA 分支了
    backbone = backbone.to(device=device, dtype=dtype).eval()
    head = PointerHead(backbone.config.hidden_size, pointer_dim=int(meta["head_dim"]))
    head.load_state_dict(meta["head"])
    head.temperature = float(meta["temperature"])
    head = head.to(device=device, dtype=torch.float32).eval()
    return Model(tok, backbone, head, device, dtype, run_dir)


def answer(model: Model, state: str, questions: dict) -> tuple[dict, int, float]:
    """
    组一个 TypeSafe 形状的响应。

    功能：noul 取 p(true)；choice 取 argmax + 每个选项的概率 + confidence；score 未实现。
    应用场景：/v1/systemone 的响应体。
    输入：state / questions — 请求内容。
    输出：(answers, input_tokens, 秒)。
    异常：编码失败 → ValueError。
    """
    # 1. 前向：一行一个问题地打分，得每个问题的选项概率
    started = time.perf_counter()
    probabilities, input_tokens, rows = model.probabilities(state, questions)
    # 2. 折成答案：noul 只取 p(true)；choice 取 argmax，confidence 按 k 等于 1 时归一到 1
    #    选项名用 encode 写进行里的 keys，不要读请求体（请求只有 criteria，没有 keys）
    answers = {}
    for row, probs in zip(rows, probabilities):
        name = row["name"]
        if row["type"] == "noul":
            answers[name] = {"type": "noul", "noul": round(float(probs[1]), 4)}
            continue
        keys = row["keys"]
        k = len(keys)
        values = {key: round(float(p), 4) for key, p in zip(keys, probs)}
        top = max(values, key=values.get)
        confidence = 1.0 if k == 1 else (values[top] - 1 / k) / (1 - 1 / k)
        answers[name] = {"type": "choice", "choice": top, "confidence": round(confidence, 4), "probabilities": values}
    # 3. 计时：只覆盖 1、2 两步，不含请求体解析与响应序列化
    return answers, input_tokens, time.perf_counter() - started
