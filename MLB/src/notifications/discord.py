# MLB/src/notifications/discord.py

from __future__ import annotations

import os
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class DiscordNotification:
    status: str
    workflow_name: str = "Daily MLB Workflow"
    message: str | None = None
    run_url: str | None = None


def get_discord_webhook_url() -> str:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL is missing")
    return webhook_url


def build_discord_payload(notification: DiscordNotification) -> dict:
    emoji = "✅" if notification.status.lower() == "success" else "🚨"

    content = f"{emoji} **{notification.workflow_name}** completed with status: `{notification.status}`"

    if notification.message:
        content += f"\n{notification.message}"

    if notification.run_url:
        content += f"\nRun: {notification.run_url}"

    return {"content": content}


def send_discord_notification(
    notification: DiscordNotification,
    webhook_url: str | None = None,
) -> None:
    webhook_url = webhook_url or get_discord_webhook_url()
    payload = build_discord_payload(notification)

    response = requests.post(webhook_url, json=payload, timeout=15)
    response.raise_for_status()