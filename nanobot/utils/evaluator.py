"""Feedback policy helpers for background message delivery."""


def should_deliver_background_message(
    *,
    feedback_level: str | None,
    severity: str = "normal",
) -> bool:
    """Return True when an explicit background message should be delivered."""
    if feedback_level == "silent":
        return False
    if feedback_level == "errors_only":
        return severity == "error"
    return True
