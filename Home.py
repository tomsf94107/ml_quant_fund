# Home.py



import streamlit as st
from datetime import datetime

st.set_page_config(page_title="📊 ML Quant Dashboard", layout="wide")

# FAST first HTML frame ↓
st.markdown("""
<div style="text-align:center; margin-top:25vh;">
  <span style="font-size:4rem;">⏳</span><br>
  <span style="font-size:1.2rem;">Loading ML-Quant&nbsp;Dashboard…</span>
</div>
""", unsafe_allow_html=True)

# Nothing else below will block the first paint
# (Optionally, after a second redirect to the Ticker page)
st.experimental_singleton(lambda: None)  # forces Streamlit to send the frame
