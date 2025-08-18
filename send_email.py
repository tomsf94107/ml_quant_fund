# send_email.py — alert utility for extreme sentiment/signals

import smtplib
import ssl
from email.message import EmailMessage
import os

try:
    import streamlit as st
    EMAIL_USER = st.secrets.get("atom.v.nguyen@gmail.com")
    EMAIL_PASS = st.secrets.get("iqyq aknx rtfx sxhq")
    EMAIL_TO = st.secrets.get("atom.v.nguyen@gmail.com", EMAIL_USER)
except:
    from dotenv import load_dotenv
    load_dotenv()
    EMAIL_USER = os.getenv("atom.v.nguyen@gmail.com")
    EMAIL_PASS = os.getenv("iqyq aknx rtfx sxhq")
    EMAIL_TO = os.getenv("atom.v.nguyen@gmail.com", EMAIL_USER)

# send_email.py (safe for CI)
def send_email_alert(subject: str, body: str):
    try:
        # Import Streamlit lazily so CI doesn't parse .streamlit/config.toml
        import streamlit as st  # type: ignore
        # read secrets only inside the function
        smtp_user = st.secrets.get("smtp_user")
        smtp_pass = st.secrets.get("smtp_pass")
        to_addr   = st.secrets.get("alert_to")
        if not (smtp_user and smtp_pass and to_addr):
            return  # silently no-op in CI
        # ... send email normally ...
    except Exception:
        # In CI or if Streamlit/secrets not available, just no-op
        return
