import streamlit as st
from fetchers import fetch_all
from nlp import extract_keywords
from summarizer import summarize_abstract
from cleaning import clean_html_abstract, normalize_authors, remove_duplicates
from utils import clean_for_biblioshiny
import pandas as pd

st.title("📚 LitFetcher: Literature Review Assistant")

query = st.text_input("Enter search query:", "quantum cryptography robotics")
max_results = st.slider("Max results", 100, 1000, 500)

if st.button("Fetch & Process"):
    with st.spinner("Fetching papers..."):
        df = fetch_all(query, save_csv=False, max_results=max_results)

    with st.spinner("Cleaning and deduplicating..."):
        df["abstract"] = df["abstract"].apply(clean_html_abstract)
        df["authors"] = df["authors"].fillna("").apply(normalize_authors)
        df = remove_duplicates(df)

    with st.spinner("Extracting keywords..."):
        df = extract_keywords(df)

    with st.spinner("Summarizing abstracts..."):
        df["summary"] = df["abstract"].apply(summarize_abstract)

    df_ready = clean_for_biblioshiny(df)

    st.success(f"✅ Done! {len(df)} papers processed.")
    st.download_button("Download CSV for Biblioshiny", df_ready.to_csv(index=False), file_name="litfetcher_output.csv")
