"""Severity evaluation for background tasks (heartbeat & cron).

After the agent executes a background task, this module makes a lightweight
LLM call to classify the result as normal or error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_EVALUATE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_notification",
            "description": "Classify a background agent result as normal or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["normal", "error"],
                        "description": "error = failures, broken workflows, or user-action-needed issues; normal = routine progress or successful non-error output",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One-sentence reason for the classification",
                    },
                },
                "required": ["severity"],
            },
        },
    }
]

_SYSTEM_PROMPT = (
    "You are a severity classifier for a background agent. "
    "You will be given the original task context and a candidate outbound message. "
    "Call the evaluate_notification tool to classify the message.\n\n"
    "Return error only when the message represents a real failure, broken workflow, "
    "or a condition the user likely needs to act on.\n\n"
    "Return normal for routine progress, successful completions, status updates, "
    "tool hints, or confirmations that everything is fine."
)


async def evaluate_response(
    response: str,
    task_context: str,
    provider: LLMProvider,
    model: str,
) -> str:
    """Classify a background-task result as normal or error.

    Uses a lightweight tool-call LLM request (same pattern as heartbeat
    ``_decide()``). Falls back to ``error`` on any failure so important
    messages are never silently dropped in errors-only mode.
    """
    try:
        llm_response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"## Task context\n{task_context}\n\n"
                    f"## Candidate message\n{response}"
                )},
            ],
            tools=_EVALUATE_TOOL,
            model=model,
            max_tokens=256,
            temperature=0.0,
        )

        if not llm_response.has_tool_calls:
            logger.warning("evaluate_response: no tool call returned, defaulting to error")
            return "error"

        args = llm_response.tool_calls[0].arguments
        severity = args.get("severity", "error")
        if severity not in {"normal", "error"}:
            severity = "error"
        reason = args.get("reason", "")
        logger.info("evaluate_response: severity={}, reason={}", severity, reason)
        return severity

    except Exception:
        logger.exception("evaluate_response failed, defaulting to error")
        return "error"
