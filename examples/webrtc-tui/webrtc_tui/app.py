from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static

from .config import RealtimeTuiConfig, load_config
from .models import ConversationUpdate
from .session import RealtimeWebRtcClient
from .state import AppStateStore


class WebRtcTuiApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #status {
        height: auto;
        padding: 0 1;
        border: solid #666666;
        margin: 0 1;
    }

    #conversation {
        height: 1fr;
        padding: 1;
        border: round #3a7a57;
        margin: 1;
    }

    #errors {
        height: 10;
        padding: 1;
        border: round #a34646;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        Binding("c", "connect_disconnect", "连接/断开"),
        Binding("m", "toggle_mic", "静音切换"),
        Binding("q", "quit", "退出"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._config: RealtimeTuiConfig | None = None
        self._client: RealtimeWebRtcClient | None = None
        self._state = AppStateStore()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(self._state.state.status, id="status")
        with Vertical(id="body"):
            yield Static("", id="conversation")
            yield Static("", id="errors")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self._config = load_config()
        except Exception as exc:
            self._state.set_status(f"配置错误：{exc}")
            self._render()
            return

        self._client = RealtimeWebRtcClient(
            config=self._config,
            on_chat=self._handle_chat,
            on_error=self._handle_error,
            on_status=self._handle_status,
        )
        self._state.set_status(
            f"未连接 | base_url={self._config.base_url} | model={self._config.model} | voice={self._config.voice}"
        )
        self._render()

    async def action_connect_disconnect(self) -> None:
        if self._client is None:
            await self._handle_error("配置尚未加载成功。")
            return
        try:
            if self._client.connected:
                await self._client.disconnect()
            else:
                await self._client.connect()
        except Exception as exc:
            await self._handle_error(str(exc))

    async def action_toggle_mic(self) -> None:
        if self._client is None:
            return
        await self._client.toggle_mute()

    async def _handle_chat(self, update: ConversationUpdate) -> None:
        self._state.apply_chat_update(update)
        self._render()

    async def _handle_error(self, text: str) -> None:
        self._state.add_error(text)
        self._render()

    async def _handle_status(self, text: str) -> None:
        self._state.set_status(text)
        self._render()

    def _render(self) -> None:
        self.query_one("#status", Static).update(self._state.state.status)
        self.query_one("#conversation", Static).update(self._state.conversation_text())
        self.query_one("#errors", Static).update(self._state.error_text())
