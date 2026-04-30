from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "webrtc-tui"))

from webrtc_tui.config import RealtimeTuiConfig  # noqa: E402
from webrtc_tui.models import ConversationUpdate, SessionRuntimeState  # noqa: E402
from webrtc_tui.session import RealtimeWebRtcClient  # noqa: E402
from webrtc_tui.state import AppStateStore  # noqa: E402


def test_app_state_store_tracks_pending_and_final_messages() -> None:
    store = AppStateStore(initial_status="init")

    store.apply_chat_update(ConversationUpdate(role="user", text="你", final=False))
    store.apply_chat_update(ConversationUpdate(role="assistant", text="好", final=False))
    store.apply_chat_update(ConversationUpdate(role="user", text="你好", final=True))
    store.apply_chat_update(ConversationUpdate(role="assistant", text="好的", final=True))

    assert store.state.pending_user == ""
    assert store.state.pending_assistant == ""
    assert [(item.role, item.text) for item in store.state.messages] == [("用户", "你好"), ("助手", "好的")]


def test_app_state_store_keeps_recent_errors_only() -> None:
    store = AppStateStore()

    for index in range(12):
        store.add_error(f"error-{index}")

    assert len(store.state.errors) == store.MAX_ERRORS
    assert store.state.errors[0] == "error-4"
    assert store.state.errors[-1] == "error-11"


def test_session_status_rendering_reflects_runtime_flags() -> None:
    async def _noop(*args, **kwargs):
        del args, kwargs

    client = RealtimeWebRtcClient(
        config=RealtimeTuiConfig(
            base_url="http://localhost:8000",
            api_key="test",
            model="gpt-realtime",
            voice="marin",
            input_device=None,
            output_device=None,
        ),
        on_chat=_noop,
        on_error=_noop,
        on_status=_noop,
    )

    client._runtime = SessionRuntimeState(
        call_id="rtc_123",
        data_channel_open=True,
        manual_mute=True,
    )
    manual = client._render_status("session.updated")

    client._runtime.manual_mute = False
    client._runtime.playback_guard_active = True
    playback_guard = client._render_status("session.updated")

    assert "call=rtc_123" in manual
    assert "dc=open" in manual
    assert "手动静音" in manual
    assert "回放保护" in playback_guard
