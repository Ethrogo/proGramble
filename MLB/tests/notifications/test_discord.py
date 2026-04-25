# MLB/tests/notifications/test_discord.py

from unittest.mock import Mock

import pytest

from notifications.discord import (
    DiscordNotification,
    build_discord_payload,
    get_discord_webhook_url,
    send_discord_notification,
)


def test_build_discord_payload_for_success_status():
    notification = DiscordNotification(
        status="success",
        workflow_name="Daily MLB Workflow",
        message="Generated 3 postable picks.",
        run_url="https://github.com/example/repo/actions/runs/123",
    )

    payload = build_discord_payload(notification)

    assert "content" in payload
    assert "✅" in payload["content"]
    assert "Daily MLB Workflow" in payload["content"]
    assert "success" in payload["content"]
    assert "Generated 3 postable picks." in payload["content"]
    assert "https://github.com/example/repo/actions/runs/123" in payload["content"]


def test_build_discord_payload_for_failure_status():
    notification = DiscordNotification(
        status="failure",
        workflow_name="Daily MLB Workflow",
        message="Daily card failed.",
        run_url="https://github.com/example/repo/actions/runs/456",
    )

    payload = build_discord_payload(notification)

    assert "content" in payload
    assert "🚨" in payload["content"]
    assert "Daily MLB Workflow" in payload["content"]
    assert "failure" in payload["content"]
    assert "Daily card failed." in payload["content"]
    assert "https://github.com/example/repo/actions/runs/456" in payload["content"]


def test_build_discord_payload_without_optional_fields():
    notification = DiscordNotification(status="success")

    payload = build_discord_payload(notification)

    assert "content" in payload
    assert "✅" in payload["content"]
    assert "Daily MLB Workflow" in payload["content"]
    assert "success" in payload["content"]


def test_get_discord_webhook_url_reads_env_var(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test")

    webhook_url = get_discord_webhook_url()

    assert webhook_url == "https://discord.com/api/webhooks/test"


def test_get_discord_webhook_url_raises_when_missing(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    with pytest.raises(ValueError, match="DISCORD_WEBHOOK_URL is missing"):
        get_discord_webhook_url()


def test_send_discord_notification_posts_payload(monkeypatch):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None

    mock_post = Mock(return_value=mock_response)
    monkeypatch.setattr("notifications.discord.requests.post", mock_post)

    notification = DiscordNotification(
        status="success",
        workflow_name="Daily MLB Workflow",
        message="Generated card successfully.",
        run_url="https://github.com/example/repo/actions/runs/789",
    )

    send_discord_notification(
        notification,
        webhook_url="https://discord.com/api/webhooks/test",
    )

    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args

    assert args[0] == "https://discord.com/api/webhooks/test"
    assert kwargs["timeout"] == 15

    payload = kwargs["json"]
    assert "Generated card successfully." in payload["content"]
    assert "success" in payload["content"]

    mock_response.raise_for_status.assert_called_once()


def test_send_discord_notification_uses_env_webhook_when_not_provided(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/from-env")

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None

    mock_post = Mock(return_value=mock_response)
    monkeypatch.setattr("notifications.discord.requests.post", mock_post)

    notification = DiscordNotification(status="success")

    send_discord_notification(notification)

    args, _ = mock_post.call_args
    assert args[0] == "https://discord.com/api/webhooks/from-env"