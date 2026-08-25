from __future__ import annotations

import httpx
import ollama

# The runner calls chat() with stream=True, so "read" here means the gap
# between chunks, not the whole generation — a real signal worth bounding,
# unlike with the old non-streaming call, where a single uniform read
# timeout would have had to tolerate an entire generation's worth of silence
# (we saw that take 10+ minutes on real dialogues) or risk killing a healthy,
# in-progress one. 15 minutes gives generous room for a slow prefill on a
# huge context plus normal inter-token gaps, while still catching a
# genuinely stalled connection (e.g. dropped by an idle-timeout somewhere on
# the path) far sooner than waiting forever.
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=10.0)


def get_client(base_url: str = "http://127.0.0.1:11434", timeout: httpx.Timeout = REQUEST_TIMEOUT) -> ollama.Client:
    return ollama.Client(host=base_url, timeout=timeout)


def list_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        client = get_client(base_url)
        response = client.list()
        return sorted(model.model for model in response.models)
    except Exception:
        return []
