from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from newme import runner


class OllamaAbortRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        runner._active_clients.clear()

    def tearDown(self) -> None:
        runner._active_clients.clear()

    def _fake_client(self, raise_on_close: bool = False) -> SimpleNamespace:
        calls: list[bool] = []

        def close():
            calls.append(True)
            if raise_on_close:
                raise RuntimeError("boom")

        return SimpleNamespace(_client=SimpleNamespace(close=close), _calls=calls)

    def test_returns_false_when_no_client_registered(self) -> None:
        self.assertFalse(runner.request_ollama_abort(999))

    def test_closes_the_registered_client_and_returns_true(self) -> None:
        client = self._fake_client()
        runner._active_clients[1] = client

        result = runner.request_ollama_abort(1)

        self.assertTrue(result)
        self.assertEqual(client._calls, [True])

    def test_only_closes_the_targeted_runs_client(self) -> None:
        client_1 = self._fake_client()
        client_2 = self._fake_client()
        runner._active_clients[1] = client_1
        runner._active_clients[2] = client_2

        runner.request_ollama_abort(1)

        self.assertEqual(client_1._calls, [True])
        self.assertEqual(client_2._calls, [])

    def test_close_failures_are_swallowed(self) -> None:
        client = self._fake_client(raise_on_close=True)
        runner._active_clients[1] = client

        result = runner.request_ollama_abort(1)

        self.assertTrue(result)
        self.assertEqual(client._calls, [True])


if __name__ == "__main__":
    unittest.main()
