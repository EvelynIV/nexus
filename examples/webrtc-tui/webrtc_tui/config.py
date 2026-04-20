from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_MODEL = "gpt-realtime"
DEFAULT_VOICE = "marin"


@dataclass(slots=True)
class RealtimeTuiConfig:
    base_url: str
    api_key: str
    model: str
    voice: str
    input_device: str | None
    output_device: str | None

    @property
    def calls_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/realtime/calls"


def load_config() -> RealtimeTuiConfig:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.getenv("REALTIME_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 REALTIME_API_KEY。请先配置 examples/webrtc-tui/.env。")

    base_url = os.getenv("REALTIME_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    model = os.getenv("REALTIME_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    voice = os.getenv("REALTIME_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
    input_device = os.getenv("REALTIME_AUDIO_INPUT_DEVICE", "").strip() or None
    output_device = os.getenv("REALTIME_AUDIO_OUTPUT_DEVICE", "").strip() or None

    return RealtimeTuiConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        voice=voice,
        input_device=input_device,
        output_device=output_device,
    )

