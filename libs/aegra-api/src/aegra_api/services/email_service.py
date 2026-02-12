"""Email Notification Service — delivers email notifications to students.

Supports SMTP delivery with HTML templates styled to match DeDataHub brand.
Each notification generates BOTH a short in-app message AND a full email
version with subject line, as required by the spec (§2.2, §3.1).
"""

from __future__ import annotations

import structlog

from aegra_api.settings import settings

logger = structlog.get_logger()


async def send_email(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: str | None = None,
) -> bool:
    """Send an email via SMTP. Returns True on success."""
    cfg = settings.email
    if not cfg.EMAIL_ENABLED or not cfg.SMTP_HOST:
        logger.debug("email_skipped", reason="email disabled or SMTP not configured")
        return False

    try:
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import aiosmtplib

        msg = MIMEMultipart("alternative")
        msg["From"] = f"{cfg.EMAIL_FROM_NAME} <{cfg.EMAIL_FROM_ADDRESS}>"
        msg["To"] = to_email
        msg["Subject"] = subject

        if text_body:
            msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        await aiosmtplib.send(
            msg,
            hostname=cfg.SMTP_HOST,
            port=cfg.SMTP_PORT,
            username=cfg.SMTP_USER or None,
            password=cfg.SMTP_PASSWORD or None,
            use_tls=cfg.SMTP_USE_TLS,
            timeout=15,
        )
        logger.info("email_sent", to=to_email, subject=subject[:60])
        return True

    except Exception as e:
        logger.error("email_send_failed", to=to_email, error=str(e))
        return False


def build_notification_email(
    student_name: str,
    title: str,
    content: str,
    action_buttons: list[dict] | None = None,
    advisor_persona: str = "Alexandra",
    category: str = "general",
) -> tuple[str, str]:
    """Build branded HTML + plain-text email from notification data.

    Returns: (html_body, text_body)
    """
    # Category-specific colors
    category_colors = {
        "deadline": "#E53E3E",
        "opportunity": "#876EFF",
        "celebration": "#38A169",
        "inactivity": "#DD6B20",
        "motivation": "#3182CE",
        "general": "#876EFF",
    }
    accent = category_colors.get(category, "#876EFF")

    # Build action buttons HTML
    buttons_html = ""
    if action_buttons:
        btns = []
        for btn in action_buttons[:3]:
            url = btn.get("url", "#")
            label = btn.get("title", "View")
            if not url.startswith("http"):
                # Make relative URLs absolute
                url = f"https://app.dedatahub.com{url}"
            btns.append(
                f'<a href="{url}" style="display:inline-block;padding:12px 24px;'
                f"background-color:{accent};color:#ffffff;text-decoration:none;"
                f'border-radius:8px;font-weight:600;margin:4px 8px 4px 0;">{label}</a>'
            )
        buttons_html = "\n".join(btns)

    # Escape content for HTML (basic)
    html_content = content.replace("\n", "<br>")

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:#f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f5f5f5;padding:32px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
        <!-- Header -->
        <tr><td style="background:linear-gradient(135deg,#876EFF,#513FB5);padding:24px 32px;">
          <h1 style="color:#ffffff;margin:0;font-size:20px;">DeDataHub</h1>
        </td></tr>
        <!-- Body -->
        <tr><td style="padding:32px;">
          <p style="color:#374151;font-size:16px;margin:0 0 8px;">Hi {student_name or "there"},</p>
          <h2 style="color:#1f2937;font-size:22px;margin:16px 0 12px;">{title}</h2>
          <div style="color:#4b5563;font-size:15px;line-height:1.6;margin:0 0 24px;">{html_content}</div>
          {f'<div style="margin:24px 0;">{buttons_html}</div>' if buttons_html else ""}
          <p style="color:#9ca3af;font-size:13px;margin:24px 0 0;border-top:1px solid #e5e7eb;padding-top:16px;">
            — {advisor_persona}, Your AI Career Advisor
          </p>
        </td></tr>
        <!-- Footer -->
        <tr><td style="background-color:#f9fafb;padding:16px 32px;text-align:center;">
          <p style="color:#9ca3af;font-size:12px;margin:0;">
            You received this because you have email notifications enabled on DeDataHub.
            <a href="https://app.dedatahub.com/dashboard/settings" style="color:#876EFF;">Manage preferences</a>
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    # Plain text version
    text_body = f"Hi {student_name or 'there'},\n\n{title}\n\n{content}\n\n— {advisor_persona}"
    if action_buttons:
        text_body += "\n\nActions:"
        for btn in action_buttons[:3]:
            url = btn.get("url", "")
            if url and not url.startswith("http"):
                url = f"https://app.dedatahub.com{url}"
            text_body += f"\n- {btn.get('title', 'View')}: {url}"

    return html_body, text_body


def build_digest_email(
    student_name: str,
    items: list[dict],
    advisor_persona: str = "Alexandra",
) -> tuple[str, str, str]:
    """Build a daily digest email bundling multiple notifications.

    Returns: (subject, html_body, text_body)
    """
    count = len(items)
    subject = f"Your Daily Career Update ({count} item{'s' if count != 1 else ''})"

    # Build items HTML
    items_html = ""
    for item in items[:10]:  # Cap at 10 items per digest
        icon = {"deadline": "📅", "opportunity": "🎯", "celebration": "🎉", "inactivity": "👋", "motivation": "💪"}.get(
            item.get("category", ""), "📌"
        )
        items_html += f"""
        <tr><td style="padding:12px 0;border-bottom:1px solid #f3f4f6;">
          <span style="font-size:18px;">{icon}</span>
          <strong style="color:#1f2937;font-size:14px;">{item.get("title", "")}</strong>
          <p style="color:#6b7280;font-size:13px;margin:4px 0 0;">{item.get("content", "")[:120]}</p>
        </td></tr>"""

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:#f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f5f5f5;padding:32px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;">
        <tr><td style="background:linear-gradient(135deg,#876EFF,#513FB5);padding:24px 32px;">
          <h1 style="color:#ffffff;margin:0;font-size:20px;">DeDataHub Daily Digest</h1>
        </td></tr>
        <tr><td style="padding:32px;">
          <p style="color:#374151;font-size:16px;">Hi {student_name or "there"}, here's your career update:</p>
          <table width="100%" cellpadding="0" cellspacing="0">{items_html}</table>
          <div style="margin:24px 0;text-align:center;">
            <a href="https://app.dedatahub.com/dashboard" style="display:inline-block;padding:12px 32px;background-color:#876EFF;color:#ffffff;text-decoration:none;border-radius:8px;font-weight:600;">Go to Dashboard</a>
          </div>
          <p style="color:#9ca3af;font-size:13px;border-top:1px solid #e5e7eb;padding-top:16px;">
            — {advisor_persona}
          </p>
        </td></tr>
        <tr><td style="background-color:#f9fafb;padding:16px 32px;text-align:center;">
          <p style="color:#9ca3af;font-size:12px;margin:0;">
            <a href="https://app.dedatahub.com/dashboard/settings" style="color:#876EFF;">Manage notification preferences</a>
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text_lines = [f"Hi {student_name or 'there'}, here's your career update:\n"]
    for item in items[:10]:
        text_lines.append(f"• {item.get('title', '')}: {item.get('content', '')[:120]}")
    text_lines.append(f"\n— {advisor_persona}")
    text_body = "\n".join(text_lines)

    return subject, html_body, text_body
