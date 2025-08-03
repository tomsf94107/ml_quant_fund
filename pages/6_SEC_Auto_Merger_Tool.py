import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="🛠 SEC Auto Merger Tool", layout="wide")

st.title("🛠 SEC Auto Merger + Cleaner")
st.markdown("Upload raw SEC `.tsv` files to auto-merge and clean them into two final outputs:")

st.markdown("""
- ✅ `Insider_Transactions_Merged_CLEAN.csv`  
- ✅ `Insider_Holdings_Merged_CLEAN.csv`
""")

# Upload files
sub_file = st.file_uploader("📄 Upload SUBMISSION.tsv", type=["tsv"], key="sub")
trans_file = st.file_uploader("📄 Upload NONDERIV_TRANS.tsv", type=["tsv"], key="trans")
hold_file = st.file_uploader("📄 Upload NONDERIV_HOLDING.tsv", type=["tsv"], key="hold")


@st.cache_data(show_spinner=False)
def clean_merge_sec_file(target_df, submission_df):
    # Standardize column names
    target_df.columns = target_df.columns.str.lower()
    submission_df.columns = submission_df.columns.str.lower()

    # Check for required column
    if "accession_number" not in target_df.columns:
        raise ValueError("❌ 'accession_number' missing in the uploaded target file.")
    if "accession_number" not in submission_df.columns:
        raise ValueError("❌ 'accession_number' missing in the uploaded submission file.")

    # Merge
    merged = target_df.merge(submission_df, on="accession_number", how="left")
    merged.dropna(axis=1, how="all", inplace=True)

    # Fill reporting owner name if missing
    if "reportingownername" in merged.columns:
        merged["reportingownername"] = merged["reportingownername"].fillna(
            merged.get("reporting_owner_name", "")
        )

    # Create filed_date
    if "formfiled" in merged.columns:
        merged["filed_date"] = pd.to_datetime(merged["formfiled"], errors='coerce')
    else:
        merged["filed_date"] = pd.NaT

    return merged


if sub_file and trans_file and hold_file:
    try:
        sub_df = pd.read_csv(sub_file, sep="\t", low_memory=False)
        trans_df = pd.read_csv(trans_file, sep="\t", low_memory=False)
        hold_df = pd.read_csv(hold_file, sep="\t", low_memory=False)

        # Merge both sets
        merged_trans = clean_merge_sec_file(trans_df, sub_df)
        merged_hold = clean_merge_sec_file(hold_df, sub_df)

        st.success("✅ Files successfully merged and cleaned!")

        # Date filter
        if not merged_trans["filed_date"].isna().all():
            start_date = st.date_input("Start Date", merged_trans["filed_date"].min().date())
            end_date = st.date_input("End Date", merged_trans["filed_date"].max().date())
            filtered_trans = merged_trans[
                (merged_trans["filed_date"] >= pd.to_datetime(start_date)) &
                (merged_trans["filed_date"] <= pd.to_datetime(end_date))
            ]
        else:
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            filtered_trans = merged_trans

        # Display preview
        st.subheader("📄 Preview: Merged Insider Transactions")
        st.dataframe(filtered_trans.head(50), use_container_width=True)

        st.subheader("📄 Preview: Merged Insider Holdings")
        st.dataframe(merged_hold.head(50), use_container_width=True)

        # Export buttons
        trans_csv = filtered_trans.to_csv(index=False).encode("utf-8")
        hold_csv = merged_hold.to_csv(index=False).encode("utf-8")

        st.download_button("⬇️ Download Insider_Transactions_Merged_CLEAN.csv", trans_csv,
                           "Insider_Transactions_Merged_CLEAN.csv", "text/csv")
        st.download_button("⬇️ Download Insider_Holdings_Merged_CLEAN.csv", hold_csv,
                           "Insider_Holdings_Merged_CLEAN.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
else:
    st.info("👆 Upload all 3 `.tsv` files above to begin.")
