"""Simple SMTP smoke test for the ai-service mailer."""

from __future__ import annotations

import argparse
import asyncio

from aegra_api.services.email_service import build_notification_email, send_email
from aegra_api.settings import settings


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Send a test email through ai-service SMTP settings.")
    parser.add_argument("--to", required=True, help="Recipient email address")
    parser.add_argument("--name", default="there", help="Recipient first name")
    args = parser.parse_args()

    cfg = settings.email
    if not cfg.EMAIL_ENABLED or not cfg.SMTP_HOST:
        print("Email is not configured. Set EMAIL_ENABLED=true and SMTP_HOST in ai-service/.env.")
        return 1

    html_body, text_body = build_notification_email(
        student_name=args.name,
        title="DeDataHub mailer smoke test",
        content="This is a direct SMTP test from ai-service. If you received this, outbound email is working.",
        action_buttons=[{"title": "Open DeDataHub", "url": "https://app.dedatahub.com/dashboard/settings"}],
        category="general",
    )

    sent = await send_email(
        to_email=args.to,
        subject="[DeDataHub] Mailer smoke test",
        html_body=html_body,
        text_body=text_body,
    )
    if not sent:
        print("Email send failed. Check SMTP_HOST/SMTP_USER/SMTP_PASSWORD and service logs.")
        return 2

    print(f"Email sent to {args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
