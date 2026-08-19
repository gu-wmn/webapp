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


class FakeClient:
    def __init__(self, effects: list):
        # Each item is either an exception instance (to raise) or a value to return.
        self._effects = list(effects)
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        effect = self._effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


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


if __name__ == "__main__":
    unittest.main()
