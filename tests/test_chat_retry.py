from __future__ import annotations

import sys
import unittest
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from newme.runner import _chat_with_retry


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChunk:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeClient:
    def __init__(self, effects: list):
        # Each item is either an exception instance (to raise) or a string —
        # the full response content, delivered as a single streamed chunk,
        # matching how _chat_with_retry now calls chat(stream=True).
        self._effects = list(effects)
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        effect = self._effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return [FakeChunk(effect)]


class MultiChunkFakeClient:
    """Simulates a real multi-chunk stream, one effect list per call."""

    def __init__(self, chunk_lists: list):
        self._chunk_lists = list(chunk_lists)
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        chunks = self._chunk_lists.pop(0)
        return (FakeChunk(c) for c in chunks)


class ChatWithRetryTests(unittest.TestCase):
    def test_succeeds_on_first_try_without_retrying(self) -> None:
        client = FakeClient(["ok"])

        result = _chat_with_retry(client, {}, retries=1, backoff_seconds=0)

        self.assertEqual(result, "ok")
        self.assertEqual(client.calls, 1)

    def test_retries_once_on_transport_error_then_succeeds(self) -> None:
        client = FakeClient([httpx.ConnectError("dropped"), "ok"])

        result = _chat_with_retry(client, {}, retries=1, backoff_seconds=0)

        self.assertEqual(result, "ok")
        self.assertEqual(client.calls, 2)

    def test_gives_up_after_exhausting_retries(self) -> None:
        client = FakeClient([httpx.ReadTimeout("timed out"), httpx.ReadTimeout("timed out again")])

        with self.assertRaises(httpx.ReadTimeout):
            _chat_with_retry(client, {}, retries=1, backoff_seconds=0)

        self.assertEqual(client.calls, 2)

    def test_non_transport_errors_are_not_retried(self) -> None:
        client = FakeClient([ValueError("bad model"), "ok"])

        with self.assertRaises(ValueError):
            _chat_with_retry(client, {}, retries=1, backoff_seconds=0)

        self.assertEqual(client.calls, 1)

    def test_requests_streaming(self) -> None:
        client = FakeClient(["ok"])

        _chat_with_retry(client, {"model": "m"}, retries=1, backoff_seconds=0)

        self.assertTrue(client.last_kwargs.get("stream"))

    def test_accumulates_multiple_chunks_into_one_string(self) -> None:
        client = MultiChunkFakeClient([["Hel", "lo", ", ", "world"]])

        result = _chat_with_retry(client, {}, retries=1, backoff_seconds=0)

        self.assertEqual(result, "Hello, world")

    def test_empty_or_none_chunk_content_is_skipped(self) -> None:
        client = MultiChunkFakeClient([["a", None, "", "b"]])

        result = _chat_with_retry(client, {}, retries=1, backoff_seconds=0)

        self.assertEqual(result, "ab")


if __name__ == "__main__":
    unittest.main()
