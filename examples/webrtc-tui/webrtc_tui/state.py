from __future__ import annotations

from .models import AppViewState, ChatLine, ConversationUpdate


class AppStateStore:
    MAX_ERRORS = 8

    def __init__(self, initial_status: str = "正在加载配置…") -> None:
        self._state = AppViewState(status=initial_status)

    @property
    def state(self) -> AppViewState:
        return self._state

    def set_status(self, text: str) -> None:
        status = text.strip()
        if status:
            self._state.status = status

    def add_error(self, text: str) -> None:
        message = text.strip()
        if not message:
            return
        self._state.errors.append(message)
        self._state.errors = self._state.errors[-self.MAX_ERRORS :]

    def apply_chat_update(self, update: ConversationUpdate) -> None:
        clean = update.text.strip()
        if update.role == "user":
            if update.final:
                if clean:
                    self._state.messages.append(ChatLine("用户", clean))
                self._state.pending_user = ""
            else:
                self._state.pending_user = clean
            return

        if update.final:
            if clean:
                self._state.messages.append(ChatLine("助手", clean))
            self._state.pending_assistant = ""
        else:
            self._state.pending_assistant = clean

    def conversation_text(self) -> str:
        lines: list[str] = []
        for item in self._state.messages:
            lines.append(f"{item.role}：{item.text}")
            lines.append("")
        if self._state.pending_user:
            lines.append(f"用户：{self._state.pending_user}")
            lines.append("")
        if self._state.pending_assistant:
            lines.append(f"助手：{self._state.pending_assistant}")
            lines.append("")
        return "\n".join(lines).strip() or "暂无对话内容。\n\n按 c 建立连接，按 m 切换麦克风静音。"

    def error_text(self) -> str:
        if not self._state.errors:
            return "错误与状态提示会显示在这里。"
        return "\n".join(self._state.errors)

