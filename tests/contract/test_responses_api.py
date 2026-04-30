from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus.api.v1.router import router
from nexus.application.container import get_container


class FakeResponsesUseCase:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    {"type": "response.created", "response": {"id": "resp_test"}},
                    {"type": "response.output_text.delta", "delta": "你好"},
                    {"type": "response.completed", "response": {"id": "resp_test"}},
                ]
            )
        return {
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "output": [],
        }


def _client(fake: FakeResponsesUseCase) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(responses=fake)
    return TestClient(app)


def test_chat_completions_route_is_not_registered() -> None:
    client = _client(FakeResponsesUseCase())

    response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 404


def test_responses_route_returns_json() -> None:
    fake = FakeResponsesUseCase()
    client = _client(fake)

    response = client.post(
        "/v1/responses",
        json={"model": "deepseek-v4-flash", "input": "你好"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "resp_test"
    assert fake.calls[0]["model"] == "deepseek-v4-flash"


def test_responses_route_streams_sse_events() -> None:
    client = _client(FakeResponsesUseCase())

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "deepseek-v4-flash", "input": "你好", "stream": True},
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "event: response.created" in body
    assert 'data: {"type": "response.output_text.delta", "delta": "你好"}' in body
