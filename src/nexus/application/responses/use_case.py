from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from nexus.infrastructure.responses import Inferencer


@dataclass
class ResponsesUseCase:
    base_url: str | None = None
    api_key: str | None = None

    def execute(
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
        inferencer = Inferencer(api_key=self.api_key, base_url=self.base_url)
        return inferencer.create(
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
