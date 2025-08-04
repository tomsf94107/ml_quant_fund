import streamlit as st
from datetime import datetime

st.set_page_config(page_title="📊 ML Quant Dashboard", layout="wide")

st.markdown("""
<div style="text-align:center; margin-top:25vh;">
  <span style="font-size:4rem;">⏳</span><br>
  <span style="font-size:1.2rem;">Loading ML-Quant&nbsp;Dashboard…</span>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def _noop():
    return None

_noop()
