"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage
from nanobot.utils.evaluator import should_deliver_background_message


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[bool | None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False
        self._feedback_level: str | None = None

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[bool | None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    @property
    def feedback_level(self) -> str | None:
        """Return the active background feedback level, if any."""
        return self._feedback_level

    def set_feedback_level(self, feedback_level: str | None) -> None:
        """Set background feedback gating for explicit message sends."""
        self._feedback_level = feedback_level

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Send a message to the user. Use this when you want to communicate something. "
            "For background runs, set severity='error' only for real failures or action-needed issues."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                },
                "severity": {
                    "type": "string",
                    "enum": ["normal", "error"],
                    "description": (
                        "Optional: classify the message for background feedback policy. "
                        "Defaults to 'normal'."
                    ),
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        severity: str = "normal",
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        if not should_deliver_background_message(
            feedback_level=self._feedback_level,
            severity=severity,
        ):
            return "Message suppressed by background feedback policy"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            },
        )

        try:
            delivered = await self._send_callback(msg)
            if delivered is not False and channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            if delivered is False:
                return "Message suppressed by background feedback policy"
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
