import streamlit as st
from send_email import send_email_alert

st.title("📊 ML Quant Fund Dashboard")
st.markdown("Welcome to the ML Quant Fund App! Use the sidebar to access different tools and dashboards.")

st.subheader("📧 Email Alert Test")

if st.button("🚨 Send Test Email"):
    send_email_alert("Test Alert", "✅ This is a test from Streamlit Cloud.")
    st.success("Test email sent! Check your inbox.")
