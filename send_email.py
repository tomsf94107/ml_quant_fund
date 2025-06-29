import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv
import os

load_dotenv()

def send_trade_email(subject, content):
    email_sender = os.getenv("atom.v.nguyen@gmail.com")
    email_password = os.getenv("iqyq aknx rtfx sxhq")
    email_receiver = os.getenv("atom.v.nguyen@gmail.com")  # send to yourself for now

    msg = EmailMessage()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    msg.set_content(content)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.send_message(msg)
