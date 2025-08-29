# app.py — Gemini Pro summaries/keywords (batched preview flow, safe assignment)
import os
import pandas as pd
import streamlit as st

from fetchers import fetch_all
from summarizer import run_summaries
from nlp import extract_keywords
from cleaning import clean_html_abstract, normalize_authors, remove_duplicates

st.set_page_config(page_title="LitFinder — Literature Review Assistant", layout="wide")
st.title("📚 LitFinder: Literature Review Assistant")
st.caption("Google Gemini Pro for fast, concise summaries & keywords. Preview-first to stay snappy.")

# ---- Health check for Gemini key ----
def _has_gemini_key() -> bool:
    try:
        _ = st.secrets["GOOGLE_API_KEY"]
        return True
    except Exception:
        return bool(os.getenv("GOOGLE_API_KEY", ""))

if not _has_gemini_key():
    st.warning(
        "Google Gemini API key not found. On Streamlit Cloud, set Settings → Secrets → "
        "`GOOGLE_API_KEY = \"your_key_here\"`. Locally, set the `GOOGLE_API_KEY` environment variable."
    )

# ---- Caching ----
@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(query: str, max_results: int):
    return fetch_all(query, max_results=max_results, save_csv=False)

@st.cache_data(show_spinner=False)
def cached_clean_html(text: str) -> str:
    return clean_html_abstract(text)

def safe_normalize_authors(series: pd.Series) -> pd.Series:
    try:
        return series.apply(normalize_authors)
    except Exception:
        return series

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if "summary" not in df.columns:
        df["summary"] = pd.Series([""] * n, dtype="object")
    if "keywords" not in df.columns:
        df["keywords"] = pd.Series([[] for _ in range(n)], dtype="object")
    return df

def clean_for_biblioshiny(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["title", "authors", "year", "abstract", "summary", "keywords", "journal"]
    out = df.copy()
    if "authors" in out.columns:
        out["authors"] = safe_normalize_authors(out["authors"])
    if "keywords" in out.columns:
        out["keywords"] = out["keywords"].apply(
            lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else (x or "")
        )
    if "abstract" in out.columns:
        out["abstract"] = out["abstract"].fillna("").astype(str).apply(cached_clean_html)
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    existing = [c for c in cols if c in out.columns]
    return out[existing] if existing else out

# ---- Sidebar ----
with st.sidebar:
    st.header("🔎 Search")
    query = st.text_input("Query", value="quantum cryptography robotics")
    max_results = st.slider("Max results", 50, 1000, 300, 50)

    st.header("🧠 NLP (Gemini Pro)")
    do_summ = st.checkbox("Generate summaries", value=True)
    do_keys = st.checkbox("Extract keywords", value=False)
    max_words = st.slider("Summary length (words)", 30, 160, 60, 5)

    st.header("⚡ Performance")
    preview_n = st.slider("Preview count (process this many now)", 10, 100, 20)
    process_all = st.toggle("Process NLP for ALL results (slower)", value=False)
    batch_size = st.slider("Batch size (API calls group size)", 5, 20, 12, 1)

    go = st.button("Search")

# ---- Main ----
if go:
    try:
        with st.spinner("Fetching papers..."):
            df = cached_fetch(query, max_results)
            if df is None or len(df) == 0:
                st.warning("No results found.")
                st.stop()

        with st.spinner("Cleaning & deduplicating..."):
            df = remove_duplicates(df).reset_index(drop=True)
            df = ensure_columns(df)
            if "abstract" not in df.columns:
                df["abstract"] = ""
            df["abstract"] = df["abstract"].fillna("").astype(str).apply(cached_clean_html)
            if "authors" in df.columns:
                df["authors"] = safe_normalize_authors(df["authors"])

        idx = df.index if process_all else df.index[:min(preview_n, len(df))]

        if do_summ:
            with st.spinner(f"Summarizing {len(idx)} abstracts..."):
                df = run_summaries(
                    df, idx, text_column="abstract", max_words=max_words, batch_size=batch_size
                )

        if do_keys:
            with st.spinner(f"Extracting keywords for {len(idx)} abstracts..."):
                df = extract_keywords(
                    df, text_column="abstract", top_n=5, idx=idx, batch_size=batch_size
                )

        df_ready = clean_for_biblioshiny(df)

        st.success(
            f"✅ Done! {len(df)} papers processed. "
            f"{'Full NLP ran.' if process_all else f'NLP preview on {len(idx)} rows.'}"
        )
        st.subheader("🔍 Preview (first 20)")
        st.dataframe(df_ready.head(20), use_container_width=True)

        st.download_button(
            "📥 Download CSV for Biblioshiny",
            df_ready.to_csv(index=False),
            file_name="litfinder_output.csv",
            mime="text/csv",
        )

        with st.expander("🧾 Columns"):
            st.markdown(
                "- **title** — paper title\n"
                "- **authors** — comma-separated list\n"
                "- **year** — publication year\n"
                "- **abstract** — cleaned abstract text\n"
                "- **summary** — Gemini Pro summary (if enabled)\n"
                "- **keywords** — Gemini Pro keyphrases (if enabled)\n"
                "- **journal** — host venue/journal (if available)\n"
            )

        if (not process_all) and (do_summ or do_keys) and len(df) > len(idx):
            st.info("Want full coverage? Re-run with “Process NLP for ALL results”.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.info("Enter a query and click **Search**. For speed, NLP runs on a small preview first.")
