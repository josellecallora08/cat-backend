"""Email service for sending password reset and notification emails."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import settings

logger = logging.getLogger(__name__)


def send_password_reset_email(to_email: str, reset_token: str) -> bool:
    """Send a password reset email with the reset link.

    Returns True if sent successfully, False otherwise.
    Always returns True in development when SMTP is not configured (logs instead).
    """
    reset_link = f"{settings.frontend_url}/login?reset_token={reset_token}"

    subject = "CATS — Reset your password"
    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 24px;">
        <div style="text-align: center; margin-bottom: 32px;">
            <h1 style="color: #2B2339; font-size: 24px; margin: 0;">CATS</h1>
            <p style="color: #666; font-size: 12px; margin: 4px 0 0;">Collection Agent Training System</p>
        </div>
        <h2 style="color: #2B2339; font-size: 18px;">Reset your password</h2>
        <p style="color: #444; font-size: 14px; line-height: 1.6;">
            We received a request to reset your password. Click the button below to create a new one.
            This link expires in {settings.reset_token_expiry_minutes} minutes.
        </p>
        <div style="text-align: center; margin: 32px 0;">
            <a href="{reset_link}"
               style="display: inline-block; background: #8F6AE0; color: #ffffff; text-decoration: none;
                      padding: 12px 32px; border-radius: 999px; font-size: 14px; font-weight: 600;">
                Reset Password
            </a>
        </div>
        <p style="color: #888; font-size: 12px; line-height: 1.5;">
            If you didn't request this, you can safely ignore this email. Your password won't change.
        </p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 32px 0;" />
        <p style="color: #aaa; font-size: 11px; text-align: center;">
            CATS — AI-powered Collection Agent Training System
        </p>
    </div>
    """

    text_body = f"""Reset your password

We received a request to reset your password. Use the link below to create a new one.
This link expires in {settings.reset_token_expiry_minutes} minutes.

{reset_link}

If you didn't request this, you can safely ignore this email.
"""

    # If SMTP is not configured, just log that a reset was attempted (NOT the link)
    if not settings.smtp_host or not settings.smtp_user:
        logger.info(
            "SMTP not configured. Password reset email would be sent to: %s",
            to_email,
        )
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_from_email}>"
        msg["To"] = to_email

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        if settings.smtp_use_tls:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port)

        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(settings.smtp_from_email, to_email, msg.as_string())
        server.quit()

        logger.info("Password reset email sent to %s", to_email)
        return True

    except Exception as e:
        logger.error("Failed to send reset email to %s: %s", to_email, str(e))
        return False
