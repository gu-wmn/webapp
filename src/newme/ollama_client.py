from __future__ import annotations

import ollama


def get_client(base_url: str = "http://127.0.0.1:11434") -> ollama.Client:
    return ollama.Client(host=base_url)


def list_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        client = get_client(base_url)
        response = client.list()
        return sorted(model.model for model in response.models)
    except Exception:
        return []
