"""Email delivery provider abstraction and development adapters."""
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, List, Dict, Any

from zenova.core.config import get_system_config
from zenova.core.logging import get_logger

logger = get_logger("zenova.auth.email")


class BaseEmailProvider(ABC):
    """Abstract interface for sending transactional account emails."""

    @abstractmethod
    async def send_verification_email(self, email: str, token: str, user_name: Optional[str] = None) -> bool:
        """Send email containing account verification link."""
        pass

    @abstractmethod
    async def send_password_reset_email(self, email: str, token: str, user_name: Optional[str] = None) -> bool:
        """Send email containing secure password reset link."""
        pass


class DevelopmentEmailProvider(BaseEmailProvider):
    """Safe local development adapter that captures emails in-memory for testing."""

    def __init__(self):
        self.sent_emails: List[Dict[str, Any]] = []

    async def send_verification_email(self, email: str, token: str, user_name: Optional[str] = None) -> bool:
        cfg = get_system_config()
        verify_url = f"{cfg.email.frontend_base_url}/verify-email?token={token}"
        record = {
            "type": "verification",
            "to": email,
            "token": token,
            "url": verify_url,
            "user_name": user_name
        }
        self.sent_emails.append(record)
        logger.info(f"[DEV EMAIL ADAPTER] Sent verification link to {email}: {verify_url}")
        return True

    async def send_password_reset_email(self, email: str, token: str, user_name: Optional[str] = None) -> bool:
        cfg = get_system_config()
        reset_url = f"{cfg.email.frontend_base_url}/reset-password?token={token}"
        record = {
            "type": "password_reset",
            "to": email,
            "token": token,
            "url": reset_url,
            "user_name": user_name
        }
        self.sent_emails.append(record)
        logger.info(f"[DEV EMAIL ADAPTER] Sent password reset link to {email}: {reset_url}")
        return True

    def get_last_token_for(self, email: str, email_type: Optional[str] = None) -> Optional[str]:
        """Test helper to retrieve the most recent token dispatched to an email."""
        for item in reversed(self.sent_emails):
            if item["to"].lower() == email.lower():
                if email_type is None or item["type"] == email_type:
                    return item["token"]
        return None

    def clear(self):
        self.sent_emails.clear()


class SMTPEmailProvider(BaseEmailProvider):
    """Standard SMTP email provider for staging and production deployments."""

    def __init__(self):
        self.cfg = get_system_config().email

    def _send_message(self, recipient: str, subject: str, html_body: str, text_body: str) -> bool:
        if not self.cfg.smtp_host:
            logger.warning("SMTP host not configured. Falling back to logger notification.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.cfg.from_email
        msg["To"] = recipient

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as server:
                if self.cfg.smtp_use_tls:
                    server.starttls()
                if self.cfg.smtp_user and self.cfg.smtp_password:
                    server.login(self.cfg.smtp_user, self.cfg.smtp_password)
                server.sendmail(self.cfg.from_email, recipient, msg.as_string())
            return True
        except Exception as exc:
            logger.error(f"Failed to deliver SMTP email to {recipient}: {exc}")
            return False

    async def send_verification_email(self, email: str, token: str, user_name: Optional[str] = None) -> bool:
        cfg = get_system_config()
        verify_url = f"{cfg.email.frontend_base_url}/verify-email?token={token}"
        greeting = f"Hello {user_name or 'there'},"
        text = f"{greeting}\n\nPlease verify your ZENOVA account by clicking: {verify_url}\nThis link expires in 24 hours."
        html = f"<p>{greeting}</p><p>Please <a href='{verify_url}'>click here to verify your ZENOVA account</a>.</p>"
        return self._send_message(email, "Verify your ZENOVA account", html, text)

    async def send_password_reset_email(self, email: str, token: str, user_name: Optional[str] = None) -> bool:
        cfg = get_system_config()
        reset_url = f"{cfg.email.frontend_base_url}/reset-password?token={token}"
        greeting = f"Hello {user_name or 'there'},"
        text = f"{greeting}\n\nWe received a request to reset your password. Reset your password here: {reset_url}\nThis link expires in 30 minutes. If you did not make this request, you can safely ignore this email."
        html = f"<p>{greeting}</p><p>Click here to <a href='{reset_url}'>reset your ZENOVA password</a>.</p><p>This link expires in 30 minutes.</p>"
        return self._send_message(email, "Reset your ZENOVA password", html, text)


# Global singleton instance
_dev_provider = DevelopmentEmailProvider()


def get_email_provider() -> BaseEmailProvider:
    """Factory returning configured email adapter."""
    cfg = get_system_config()
    if cfg.email.provider == "smtp" and cfg.email.smtp_host:
        return SMTPEmailProvider()
    return _dev_provider
