from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


def _effective_api_key(api_key: str | None, base_url: str | None) -> str | None:
    if api_key:
        return api_key
    if base_url:
        return "no-key"
    return None


class Inferencer:
    """Responses API 推理器。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.client = OpenAI(api_key=_effective_api_key(api_key, base_url), base_url=base_url)

    def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        instructions: str | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        stream: bool = False,
        store: bool = True,
    ):
        try:
            payload = _create_response_payload(
                model=model,
                input=input,
                instructions=instructions,
                tools=tools,
                previous_response_id=previous_response_id,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stream=stream,
                store=store,
            )
            return self.client.responses.create(**payload)
        except Exception as err:
            logger.error("Responses inference error: %s", err)
            raise

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()


class AsyncInferencer:
    """异步 Responses API 推理器。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=_effective_api_key(api_key, base_url), base_url=base_url)

    async def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        instructions: str | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        stream: bool = False,
        store: bool = True,
    ):
        try:
            payload = _create_response_payload(
                model=model,
                input=input,
                instructions=instructions,
                tools=tools,
                previous_response_id=previous_response_id,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stream=stream,
                store=store,
            )
            return await self.client.responses.create(**payload)
        except Exception as err:
            logger.error("Async responses inference error: %s", err)
            raise

    async def close(self) -> None:
        if hasattr(self.client, "close"):
            await self.client.close()


def _create_response_payload(
    *,
    model: str,
    input: str | list[dict[str, Any]],
    instructions: str | None,
    tools: Iterable[dict[str, Any]] | None,
    previous_response_id: str | None,
    temperature: float | None,
    max_output_tokens: int | None,
    stream: bool,
    store: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input,
        "stream": stream,
        "store": store,
    }
    if instructions is not None:
        payload["instructions"] = instructions
    if tools is not None:
        payload["tools"] = list(tools)
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    if temperature is not None:
        payload["temperature"] = temperature
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    return payload
