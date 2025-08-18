#!/usr/bin/env python3
# send_email.py — CI-safe alert utility (no Streamlit at import time)

from __future__ import annotations
import os, smtplib, ssl
from email.message import EmailMessage

def _load_creds():
    # 1) Prefer environment variables (CI-friendly)
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    alert_to  = os.getenv("ALERT_TO")
    from_addr = os.getenv("SMTP_FROM") or smtp_user

    # 2) If running inside Streamlit, allow secrets-based override (lazy import)
    try:
        import streamlit as st  # only if available at runtime
        # support both flat keys and a [smtp] table
        def get(k, default=None):
            v = st.secrets.get(k, default)
            if isinstance(v, dict):
                return v
            return v

        smtp_tbl = get("smtp", {}) or {}
        smtp_user = smtp_user or get("smtp_user") or smtp_tbl.get("user")
        smtp_pass = smtp_pass or get("smtp_pass") or smtp_tbl.get("pass")
        smtp_host = smtp_host or get("smtp_host") or smtp_tbl.get("host")
        smtp_port = smtp_port or get("smtp_port") or smtp_tbl.get("port")
        alert_to  = alert_to  or get("alert_to")  or smtp_tbl.get("alert_to")
        from_addr = from_addr or get("smtp_from") or smtp_tbl.get("from") or smtp_user
    except Exception:
        pass

    # 3) Defaults (Gmail)
    smtp_host = smtp_host or "smtp.gmail.com"
    smtp_port = int(smtp_port or 465)

    return smtp_host, smtp_port, smtp_user, smtp_pass, alert_to, from_addr

def send_email_alert(subject: str, body: str, to: str | None = None) -> bool:
    """
    Returns True on success, False on no-op/failure.
    Works in Streamlit (secrets) or CI (env).
    """
    smtp_host, smtp_port, smtp_user, smtp_pass, alert_to, from_addr = _load_creds()
    to_addr = to or alert_to
    # If creds are missing (e.g., CI without secrets), silently no-op.
    if not (smtp_user and smtp_pass and to_addr):
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ssl.create_default_context()) as s:
                s.login(smtp_user, smtp_pass)
                s.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(smtp_user, smtp_pass)
                s.send_message(msg)
        return True
    except Exception:
        # fail quietly; callers don't rely on email side effects
        return False
