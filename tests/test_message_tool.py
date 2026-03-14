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


@pytest.mark.asyncio
async def test_message_tool_errors_only_suppresses_normal_messages() -> None:
    async def _send(_msg) -> bool:
        raise AssertionError("suppressed message should not be sent")

    tool = MessageTool(send_callback=_send, default_channel="telegram", default_chat_id="chat1")
    tool.begin_background_delivery("errors_only")

    result = await tool.execute(content="routine status")

    assert result == "Message suppressed by background feedback policy"
    assert tool._sent_in_turn is False


@pytest.mark.asyncio
async def test_message_tool_errors_only_allows_error_messages() -> None:
    sent = []

    async def _send(msg) -> bool:
        sent.append(msg)
        return True

    tool = MessageTool(send_callback=_send, default_channel="telegram", default_chat_id="chat1")
    tool.begin_background_delivery("errors_only")

    result = await tool.execute(content="disk full", severity="error")

    assert result == "Message sent to telegram:chat1"
    assert len(sent) == 1
    assert tool._sent_in_turn is True


@pytest.mark.asyncio
async def test_message_tool_silent_suppresses_error_messages() -> None:
    async def _send(_msg) -> bool:
        raise AssertionError("suppressed message should not be sent")

    tool = MessageTool(send_callback=_send, default_channel="telegram", default_chat_id="chat1")
    tool.begin_background_delivery("silent")

    result = await tool.execute(content="disk full", severity="error")

    assert result == "Message suppressed by background feedback policy"
    assert tool._sent_in_turn is False
