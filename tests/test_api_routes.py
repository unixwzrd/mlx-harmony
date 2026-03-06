"""Unit tests for reusable API route registration helpers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_harmony.api_routes import (
    register_chat_completions_route,
    register_placeholder_post_endpoints,
)


def test_register_chat_completions_route_wires_handler() -> None:
    """Register chat-completions route using the provided handler callable."""
    app = FastAPI()

    async def _handler() -> dict[str, str]:
        return {"id": "chatcmpl-test"}

    register_chat_completions_route(app=app, handler=_handler)
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 200
    assert response.json() == {"id": "chatcmpl-test"}


def test_register_placeholder_post_endpoints_returns_501_payloads() -> None:
    """Return standardized not-implemented payloads for placeholder routes."""
    app = FastAPI()
    register_placeholder_post_endpoints(app=app, endpoints=["/v1/embeddings"])
    client = TestClient(app)
    response = client.post("/v1/embeddings")
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["type"] == "not_implemented_error"
    assert body["error"]["code"] == "not_implemented"

