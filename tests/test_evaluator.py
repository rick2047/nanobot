from nanobot.utils.evaluator import should_deliver_background_message


def test_all_level_delivers_normal_messages() -> None:
    assert should_deliver_background_message(
        feedback_level="all",
        severity="normal",
    ) is True


def test_errors_only_suppresses_normal_messages() -> None:
    assert should_deliver_background_message(
        feedback_level="errors_only",
        severity="normal",
    ) is False


def test_errors_only_delivers_error_messages() -> None:
    assert should_deliver_background_message(
        feedback_level="errors_only",
        severity="error",
    ) is True


def test_silent_suppresses_everything() -> None:
    assert should_deliver_background_message(
        feedback_level="silent",
        severity="error",
    ) is False
