# -*- coding: utf-8 -*-
"""
DeepSeek 非流式 HTTP。

使用前注意事项：只发 stream=false、思考关闭、不带 tools。地址、模型名、密钥只读进程环境
（CHAT_API_URL、CHAT_MODEL_NAME、DEEPSEEK_API_KEY），缺一就不发请求。本仓由
backend.conf.apply_env 先把根目录 .env 灌进环境，这里不再读一份 .env，也不另写默认地址或模型名。不另开端口。
kind 按本类实际发出的非流式返回来分：stop、length、非 200 的 error。思考、工具调用、流式不发，也不为它们分支。
对外：HttpClient、ChatReply、ReplyKind。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum

import requests


class ReplyKind(StrEnum):
    """一次非流式调用的返回类别。"""

    STOP = "stop"
    LENGTH = "length"
    JSON_INVALID = "json_invalid"
    HTTP_ERROR = "http_error"
    TRANSPORT = "transport"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True)
class ChatReply:
    """
    一次非流式补全的分类结果。

    stop
        HTTP 200，choices[0].finish_reason 为 stop，message.content 是字符串。
        json_object 时 content 是一个 JSON 对象，放进 value。
    length
        HTTP 200，finish_reason 为 length，content 是被截断的字符串。
    json_invalid
        要求了 JSON 对象且 finish_reason 为 stop，但 content 不能 json.loads 成对象。
        这是对 content 字符串的解析，不是另一种 HTTP 正文。
    http_error
        HTTP 不是 200。正文是 {error: {message, type, param, code}}。
    transport
        请求没拿到 HTTP 响应（超时、连接失败）。不是接口正文。
    not_configured
        CHAT_API_URL、CHAT_MODEL_NAME、DEEPSEEK_API_KEY 有空的，请求没发出。
    """

    kind: ReplyKind
    content: str = ""
    status_code: int = 0
    error_message: str = ""
    value: dict | None = None


class HttpClient:
    """非流式 DeepSeek Chat Completions。各模块用这个类发请求，不自己组 HTTP。"""

    TIMEOUT_SEC = 120
    MAX_TOKENS = 2048

    def _from_body(self, status_code: int, body: object, *, json_object: bool) -> ChatReply:
        """把 HTTP 状态和正文收成一种 ChatReply。"""
        if status_code != 200:
            message = ""
            if isinstance(body, dict):
                message = str(body["error"]["message"])
            return ChatReply(
                kind=ReplyKind.HTTP_ERROR,
                status_code=status_code,
                error_message=message,
            )
        choice = body["choices"][0]
        content = choice["message"]["content"]
        finish = choice["finish_reason"]
        if finish == "length":
            return ChatReply(kind=ReplyKind.LENGTH, content=content, status_code=200)
        if json_object:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                return ChatReply(
                    kind=ReplyKind.JSON_INVALID,
                    content=content,
                    status_code=200,
                    error_message=str(exc),
                )
            if not isinstance(parsed, dict):
                return ChatReply(
                    kind=ReplyKind.JSON_INVALID,
                    content=content,
                    status_code=200,
                    error_message="content 不是 JSON 对象",
                )
            return ChatReply(
                kind=ReplyKind.STOP,
                content=content,
                status_code=200,
                value=parsed,
            )
        return ChatReply(kind=ReplyKind.STOP, content=content, status_code=200)

    def complete(self, prompt: str, *, json_object: bool = False) -> ChatReply:
        """
        发一次非流式补全，按返回分类。

        功能：思考关闭；json_object 为真时带 response_format json_object。
        应用场景：意图分类、改写、在线抽标签、入库抽标签、闲聊、按命中生成。
        输入：prompt — 一条 user 正文；json_object — 是否要求一个 JSON 对象。
        输出：ChatReply。成功的 JSON 对象在 value；失败也在 kind 里，不抛。
        异常：不抛。环境变量为空、超时、非 200，都是一种 kind。
        """
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        url = os.getenv("CHAT_API_URL", "").strip()
        model = os.getenv("CHAT_MODEL_NAME", "").strip()
        missing = [
            name
            for name, value in (
                ("DEEPSEEK_API_KEY", api_key),
                ("CHAT_API_URL", url),
                ("CHAT_MODEL_NAME", model),
            )
            if not value
        ]
        if missing:
            return ChatReply(
                kind=ReplyKind.NOT_CONFIGURED,
                error_message="未配置 " + "、".join(missing),
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": self.MAX_TOKENS,
            "thinking": {
                "type": "disabled",
            },
        }
        if json_object:
            body["response_format"] = {
                "type": "json_object",
            }
        try:
            response = requests.post(url, headers=headers, json=body, timeout=self.TIMEOUT_SEC)
        except requests.RequestException as exc:
            return ChatReply(
                kind=ReplyKind.TRANSPORT,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return self._from_body(response.status_code, payload, json_object=json_object)
