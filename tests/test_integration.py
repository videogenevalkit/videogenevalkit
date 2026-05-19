"""End-to-end integration tests with mocked HTTP endpoints.

Spawns an aiohttp test server that emulates an OpenAI-compatible
/chat/completions endpoint, then exercises the full pipeline including
HTTPDispatcher retry, ApiCallLogger, frame extraction, and aggregation.

These tests don't need vLLM or any real model weights.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeOpenAIServer:
    """Tiny HTTP/1.1 server that answers /v1/chat/completions with a canned JSON.

    The canned response can be customized per test via `set_response`.
    Tracks every request body it received so the test can assert.
    """

    def __init__(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.port = _free_port()
        self.responses: list[dict] = []        # rotating queue of canned responses
        self.requests: list[dict] = []
        self._error_codes: list[int] = []      # if set, returned before responses

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a, **k):    # silence
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    payload = {"_raw": body}
                outer.requests.append(payload)
                # Possibly fail this request first to exercise retries.
                if outer._error_codes:
                    code = outer._error_codes.pop(0)
                    err_body = json.dumps({"error": "transient"}).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(err_body)
                    return
                resp = outer.responses.pop(0) if outer.responses else {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "model": payload.get("model", "fake"),
                }
                body_out = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_out)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body_out)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        # give the server a moment to bind
        time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()

    def queue_response(self, payload: dict) -> None:
        self.responses.append(payload)

    def queue_error(self, status: int) -> None:
        self._error_codes.append(status)


def test_openai_compatible_judge_sync_call_logs():
    """Sync chat_text round-trips through the fake server and is api_log'd."""
    from videvalkit.scorers.vlm_judge.openai_compatible import OpenAICompatibleVLMJudge
    from videvalkit.storage import Workspace, ApiCallLogger

    with FakeOpenAIServer() as srv, tempfile.TemporaryDirectory() as td:
        srv.queue_response({
            "choices": [{"message": {"content": '{"score": 4}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            "model": "fake-model",
        })
        ws = Workspace(td)
        logger = ApiCallLogger(ws.layout, provider="fake", model="fake-model")
        judge = OpenAICompatibleVLMJudge(
            name="fake", endpoint=srv.url, model="fake-model",
            provider="fake", logger=logger,
        )
        resp = judge.chat_text(user="hello", max_tokens=10)
        assert resp["content"] == '{"score": 4}'
        assert resp["usage"]["total_tokens"] == 15
        # api_log was written
        calls = list(ws.layout.api_calls_dir.rglob("*.jsonl"))
        assert len(calls) == 1
        rec = json.loads(calls[0].read_text().strip())
        assert rec["model"] == "fake-model"
        assert rec["response"]["response"] == '{"score": 4}'
        assert srv.requests[0]["model"] == "fake-model"


def test_http_dispatcher_retries_on_429():
    """One 429 followed by a 200 → caller still gets the 200, with logged warning."""
    from videvalkit.scheduler.http_pool import HTTPDispatcher

    async def _go():
        with FakeOpenAIServer() as srv:
            srv.queue_error(429)
            srv.queue_response({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {}, "model": "fake",
            })
            d = HTTPDispatcher(concurrency=1, max_retries=3, timeout_s=5)
            try:
                resp = await d.post_chat(
                    endpoint=srv.url,
                    payload={"model": "fake", "messages": [{"role": "user", "content": "x"}]},
                    provider="fake",
                )
                assert resp["choices"][0]["message"]["content"] == "ok"
                assert len(srv.requests) == 2
            finally:
                await d.aclose()
    asyncio.run(_go())


def test_cross_benchmark_combine_summaries():
    """combine_summaries z-scores per benchmark and produces a unified ranking."""
    from videvalkit.aggregators import combine_summaries
    from videvalkit.core.types import Summary

    summaries = [
        Summary(benchmark="vbench", model="A", per_dimension={}, overall=0.80, n_videos=10, n_prompts=5, aggregator="ws"),
        Summary(benchmark="vbench", model="B", per_dimension={}, overall=0.60, n_videos=10, n_prompts=5, aggregator="ws"),
        Summary(benchmark="worldjen", model="A", per_dimension={}, overall=4.5, n_videos=10, n_prompts=5, aggregator="phas"),
        Summary(benchmark="worldjen", model="B", per_dimension={}, overall=3.0, n_videos=10, n_prompts=5, aggregator="phas"),
        Summary(benchmark="videobench", model="A", per_dimension={}, overall=4.0, n_videos=10, n_prompts=5, aggregator="ws"),
        Summary(benchmark="videobench", model="B", per_dimension={}, overall=3.5, n_videos=10, n_prompts=5, aggregator="ws"),
    ]
    rep = combine_summaries(summaries)
    # A wins in every benchmark, so unified z must rank A first.
    assert rep["ranking"][0]["model"] == "A"
    assert rep["ranking"][1]["model"] == "B"
    assert rep["unified"]["A"] > rep["unified"]["B"]
    # per-bench captured
    assert set(rep["per_benchmark"]) == {"vbench", "worldjen", "videobench"}
    # BT ratings present
    assert "bt" in rep and rep["bt"]["point"]["A"] > rep["bt"]["point"]["B"]
