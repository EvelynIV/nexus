from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexus.infrastructure.chat.inferencer import AsyncInferencer, Inferencer


@patch("nexus.infrastructure.chat.inferencer.OpenAI")
def test_sync_inferencer_does_not_special_case_qwen35(mock_openai) -> None:
    create_mock = Mock(return_value=object())
    mock_openai.return_value = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_mock),
        )
    )
    inferencer = Inferencer(api_key="dummy", base_url="http://localhost:11434/v1")

    inferencer.chat(
        messages=[{"role": "user", "content": "你好"}],
        model="qwen3.5:9b",
        stream=True,
    )

    assert "reasoning_effort" not in create_mock.call_args.kwargs


@patch("nexus.infrastructure.chat.inferencer.OpenAI")
def test_sync_inferencer_does_not_special_case_qwen35_case_insensitively(mock_openai) -> None:
    create_mock = Mock(return_value=object())
    mock_openai.return_value = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_mock),
        )
    )
    inferencer = Inferencer(api_key="dummy", base_url="http://localhost:11434/v1")

    inferencer.chat(
        messages=[{"role": "user", "content": "你好"}],
        model="QWEN3.5:9b",
    )

    assert "reasoning_effort" not in create_mock.call_args.kwargs


@patch("nexus.infrastructure.chat.inferencer.OpenAI")
def test_sync_inferencer_skips_reasoning_effort_for_other_models(mock_openai) -> None:
    create_mock = Mock(return_value=object())
    mock_openai.return_value = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_mock),
        )
    )
    inferencer = Inferencer(api_key="dummy", base_url="http://localhost:11434/v1")

    inferencer.chat(
        messages=[{"role": "user", "content": "你好"}],
        model="gpt-4o-mini",
    )

    assert "reasoning_effort" not in create_mock.call_args.kwargs


@patch("nexus.infrastructure.chat.inferencer.AsyncOpenAI")
@pytest.mark.asyncio
async def test_async_inferencer_does_not_special_case_qwen35(mock_async_openai) -> None:
    create_mock = AsyncMock(return_value=object())
    mock_async_openai.return_value = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_mock),
        )
    )
    inferencer = AsyncInferencer(api_key="dummy", base_url="http://localhost:11434/v1")

    await inferencer.chat(
        messages=[{"role": "user", "content": "你好"}],
        model="qwen3.5:9b",
        stream=True,
    )

    assert "reasoning_effort" not in create_mock.call_args.kwargs


@patch("nexus.infrastructure.chat.inferencer.AsyncOpenAI")
@pytest.mark.asyncio
async def test_async_inferencer_skips_reasoning_effort_for_other_models(mock_async_openai) -> None:
    create_mock = AsyncMock(return_value=object())
    mock_async_openai.return_value = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_mock),
        )
    )
    inferencer = AsyncInferencer(api_key="dummy", base_url="http://localhost:11434/v1")

    await inferencer.chat(
        messages=[{"role": "user", "content": "你好"}],
        model="gpt-4o-mini",
    )

    assert "reasoning_effort" not in create_mock.call_args.kwargs
