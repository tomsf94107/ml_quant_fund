import streamlit as st

st.title("🔐 Secret Debugger")
st.write("Secrets:", st.secrets["gcp_service_account"])
