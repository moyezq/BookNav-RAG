"""DeepSeek Chat Completions 的唯一调用方。不是本机进程，不进 start.py。"""

from service.deepseek.http_client import ChatReply, HttpClient, ReplyKind

__all__ = ["ChatReply", "HttpClient", "ReplyKind"]
