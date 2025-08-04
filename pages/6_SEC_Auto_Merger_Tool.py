import streamlit as st
import pandas as pd
import io
import zipfile
import os
import json

st.set_page_config(page_title="🗂 SEC Auto-Merger Tool", layout="wide")
st.title("🗂 SEC Auto-Merger + Cleaner Tool")
st.caption("Upload raw SEC `.tsv` files — we'll merge, clean, and enrich them with sector + ticker info.")

@st.cache_data(show_spinner=False)
def load_cik_to_ticker():
    with open("data/cik_to_ticker.json") as f:
        data = json.load(f)
    return {str(v["cik_str"]).zfill(10): v["ticker"] for v in data.values()}

@st.cache_data(show_spinner=False)
def load_sector_mapping():
    df = pd.read_csv("data/ticker_sector_industry.csv")
    return df.set_index("ticker").to_dict(orient="index")


@st.cache_data(show_spinner=False)
def clean_merge_sec_file(target_df, submission_df, cik_map, sector_map):
    # Merge
    if "ACCESSION_NUMBER" not in target_df.columns or "ACCESSION_NUMBER" not in submission_df.columns:
        st.error("❌ 'ACCESSION_NUMBER' column missing in one of the uploaded files.")
        st.stop()

    merged = target_df.merge(submission_df, on="ACCESSION_NUMBER", how="left")

    # CIK → Ticker
    merged["cik"] = merged["ISSUERCIK"].astype(str).str.zfill(10)
    merged["ticker"] = merged["cik"].map(cik_map)

    # Add sector/industry
    merged["sector"] = merged["ticker"].map(lambda t: sector_map.get(t, {}).get("sector"))
    merged["industry"] = merged["ticker"].map(lambda t: sector_map.get(t, {}).get("industry"))

    # Parse date
    merged["filed_date"] = pd.to_datetime(merged["FILING_DATE"], errors="coerce")

    # Clean up
    merged.dropna(axis=1, how="all", inplace=True)
    return merged

uploaded_files = st.file_uploader("📥 Upload your SEC files (.tsv):", type=["tsv"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        file_dict = {f.name.lower(): f for f in uploaded_files}
        sub_file = file_dict.get("submission.tsv")
        trans_file = file_dict.get("nonderiv_trans.tsv")
        hold_file = file_dict.get("nonderiv_holding.tsv")

        if not (sub_file and trans_file and hold_file):
            st.error("Please upload all 3 files: `SUBMISSION.tsv`, `NONDERIV_TRANS.tsv`, and `NONDERIV_HOLDING.tsv`.")
        else:
            sub_df = pd.read_csv(sub_file, sep="\t", low_memory=False)
            trans_df = pd.read_csv(trans_file, sep="\t", low_memory=False)
            hold_df = pd.read_csv(hold_file, sep="\t", low_memory=False)

            cik_map = load_cik_to_ticker()
            sector_map = load_sector_mapping()

            merged_trans = clean_merge_sec_file(trans_df, sub_df, cik_map, sector_map)
            merged_hold = clean_merge_sec_file(hold_df, sub_df, cik_map, sector_map)

            st.success("✅ Files successfully merged and cleaned!")

            # Filter
            tickers = sorted(merged_trans["ticker"].dropna().unique())
            with st.expander("🔍 Optional: Filter Results"):
                selected_ticker = st.selectbox("Filter by Ticker", ["All"] + tickers)
                start_date = st.date_input("Start Date", merged_trans["filed_date"].min().date())
                end_date = st.date_input("End Date", merged_trans["filed_date"].max().date())

                def apply_filter(df):
                    df = df[df["filed_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
                    if selected_ticker != "All":
                        df = df[df["ticker"] == selected_ticker]
                    return df

                merged_trans = apply_filter(merged_trans)
                merged_hold = apply_filter(merged_hold)

            # Display
            st.subheader("📄 Cleaned Transactions")
            st.dataframe(merged_trans)

            st.subheader("📄 Cleaned Holdings")
            st.dataframe(merged_hold)

            # Download
            st.download_button("📤 Download Transactions CSV", merged_trans.to_csv(index=False), file_name="Insider_Transactions_Merged_CLEAN.csv")
            st.download_button("📤 Download Holdings CSV", merged_hold.to_csv(index=False), file_name="Insider_Holdings_Merged_CLEAN.csv")
