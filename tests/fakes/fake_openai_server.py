"""Fake in-process OpenAI-compatible server for hermetic testing of LLMClient."""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional
import httpx


class FakeOpenAIServer:
    """Scriptable in-process OpenAI-compatible mock server via httpx transport."""

    def __init__(self):
        self.responses: List[Any] = []
        self.recorded_requests: List[Dict[str, Any]] = []
        self.delay: float = 0.0
        self.error_status: Optional[int] = None
        self.refuse_connection: bool = False

    def enqueue_error(self, status_code: int = 500, error_message: str = "Fake server error"):
        """Queue an HTTP error response."""
        self.responses.append({"_is_error": True, "status_code": status_code, "message": error_message})

    def enqueue_response(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None, finish_reason: str = "stop"):
        """Queue a scripted response."""
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.responses.append({
            "id": f"fake-cmpl-{len(self.recorded_requests)}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25,
            }
        })

    def enqueue_stream_tokens(self, tokens: List[str]):
        """Queue chunks for a streaming request."""
        chunks = []
        for i, tok in enumerate(tokens):
            chunks.append({
                "id": "fake-stream",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "fake-model",
                "choices": [{
                    "index": 0,
                    "delta": {"content": tok},
                    "finish_reason": "stop" if i == len(tokens) - 1 else None,
                }]
            })
        self.responses.append({"_is_stream": True, "chunks": chunks})

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Synchronous/async httpx request handler."""
        if self.refuse_connection:
            raise httpx.ConnectError("Connection refused by fake server")

        req_body = json.loads(request.read().decode("utf-8")) if request.content else {}
        self.recorded_requests.append({
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": req_body,
        })

        if self.error_status:
            return httpx.Response(self.error_status, text=f"HTTP {self.error_status} error from fake server")

        if not self.responses:
            # Default fallback response
            return httpx.Response(
                200,
                json={
                    "id": "fake-cmpl-default",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "I am completely operational, Dave."},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                }
            )

        resp_data = self.responses.pop(0)

        if isinstance(resp_data, dict) and resp_data.get("_is_error"):
            code = resp_data.get("status_code", 500)
            msg = resp_data.get("message", "Error")
            return httpx.Response(code, text=msg)

        if isinstance(resp_data, dict) and resp_data.get("_is_stream"):
            # Format SSE lines
            lines = []
            for chunk in resp_data["chunks"]:
                lines.append(f"data: {json.dumps(chunk)}\n\n")
            lines.append("data: [DONE]\n\n")
            sse_content = "".join(lines).encode("utf-8")
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse_content)

        return httpx.Response(200, json=resp_data)

    def get_transport(self) -> httpx.MockTransport:
        """Returns an httpx MockTransport backed by this fake server."""
        return httpx.MockTransport(self.handle_request)
