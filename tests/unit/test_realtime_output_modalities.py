from __future__ import annotations

import pytest

from nexus.application.realtime.service import RealtimeApplicationService


def _service_without_init() -> RealtimeApplicationService:
    return RealtimeApplicationService.__new__(RealtimeApplicationService)


def test_normalize_output_modalities_rejects_dual_modalities() -> None:
    service = _service_without_init()

    with pytest.raises(ValueError, match="not allowed"):
        service._normalize_output_modalities(["audio", "text"])


def test_normalize_output_modalities_accepts_single_audio_or_text() -> None:
    service = _service_without_init()

    assert service._normalize_output_modalities(["audio"]) == ["audio"]
    assert service._normalize_output_modalities(["text"]) == ["text"]
