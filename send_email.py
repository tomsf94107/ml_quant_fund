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

def send_email_alert(subject: str, body: str, to: str = EMAIL_TO):
    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        print("✅ Email sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")