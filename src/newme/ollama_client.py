from __future__ import annotations

import httpx
import ollama

# ollama.Client's .chat() defaults to stream=False, so it blocks for the
# *entire* response in one go — Ollama sends nothing back until generation is
# fully finished. That means a single uniform timeout is wrong: connecting to
# the host should be fast no matter what, but waiting for the model to finish
# thinking can legitimately take a long time on a large model with
# schema-constrained output, and there's no fixed number that's "long enough"
# across every model/prompt/hardware combination. A blanket timeout there
# doesn't just fail to help — it actively kills healthy, in-progress
# generations and misreports it as a lost connection.
#
# So only the phases that really should be fast get a bound. Reading (i.e.
# waiting for the model) is deliberately unbounded — a request that's
# genuinely stuck, as opposed to just slow, is what the Abort button and its
# connection-close are for; a human judging "has this gone on too long" beats
# a guessed timeout.
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)


def get_client(base_url: str = "http://127.0.0.1:11434", timeout: httpx.Timeout = REQUEST_TIMEOUT) -> ollama.Client:
    return ollama.Client(host=base_url, timeout=timeout)


def list_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        client = get_client(base_url)
        response = client.list()
        return sorted(model.model for model in response.models)
    except Exception:
        return []
