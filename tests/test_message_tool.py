import pytest

from nanobot.agent.tools.message import MessageTool


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_suppressed_send_does_not_mark_turn_as_sent() -> None:
    async def _suppress(_msg) -> bool:
        return False

    tool = MessageTool(send_callback=_suppress, default_channel="telegram", default_chat_id="chat1")

    result = await tool.execute(content="test")

    assert result == "Message suppressed by background feedback policy"
    assert tool._sent_in_turn is False
