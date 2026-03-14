import pytest

from nanobot.utils.evaluator import evaluate_response
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class DummyProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)

    async def chat(self, *args, **kwargs) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


def _eval_tool_call(severity: str, reason: str = "") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                id="eval_1",
                name="evaluate_notification",
                arguments={"severity": severity, "reason": reason},
            )
        ],
    )


@pytest.mark.asyncio
async def test_classifies_error() -> None:
    provider = DummyProvider([_eval_tool_call("error", "user action required")])
    result = await evaluate_response("Task completed with results", "check emails", provider, "m")
    assert result == "error"


@pytest.mark.asyncio
async def test_classifies_normal() -> None:
    provider = DummyProvider([_eval_tool_call("normal", "routine check, nothing new")])
    result = await evaluate_response("All clear, no updates", "check status", provider, "m")
    assert result == "normal"


@pytest.mark.asyncio
async def test_fallback_on_error() -> None:
    class FailingProvider(DummyProvider):
        async def chat(self, *args, **kwargs) -> LLMResponse:
            raise RuntimeError("provider down")

    provider = FailingProvider([])
    result = await evaluate_response("some response", "some task", provider, "m")
    assert result == "error"


@pytest.mark.asyncio
async def test_no_tool_call_fallback() -> None:
    provider = DummyProvider([LLMResponse(content="I think you should notify", tool_calls=[])])
    result = await evaluate_response("some response", "some task", provider, "m")
    assert result == "error"
