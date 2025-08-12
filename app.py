# app.py
import os
import streamlit as st
import pandas as pd

from fetchers import fetch_all
from summarizer import run_summaries
from nlp import extract_keywords
from cleaning import clean_html_abstract, normalize_authors, remove_duplicates

# =========================
# Streamlit page setup
# =========================
st.set_page_config(page_title="LitFinder — Literature Review Assistant", layout="wide")
st.title("📚 LitFinder: Literature Review Assistant")
st.caption("Fast literature search with Gemini summaries & keywords. NLP runs on a preview first for speed.")

# =========================
# Small health check (API key present?)
# =========================
def _has_gemini_key() -> bool:
    try:
        _ = st.secrets["GOOGLE_API_KEY"]
        return True
    except Exception:
        return bool(os.getenv("GOOGLE_API_KEY", ""))

if not _has_gemini_key():
    st.warning(
        "Gemini API key not found. On Streamlit Cloud, set Settings → Secrets → "
        "`GOOGLE_API_KEY = \"your_api_key\"`. Locally, you can set the `GOOGLE_API_KEY` env var."
    )

# =========================
# Caching helpers
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(query: str, max_results: int):
    """Fetch results from APIs (OpenAlex/Crossref/optional arXiv)."""
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
    for col in ["summary", "keywords"]:
        if col not in df.columns:
            df[col] = ""
    return df

def clean_for_biblioshiny(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a compact set of columns suitable for Biblioshiny import.
    Creates columns if missing and normalizes types.
    """
    columns_to_keep = ["title", "authors", "year", "abstract", "summary", "keywords", "journal"]
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

    existing = [c for c in columns_to_keep if c in out.columns]
    return out[existing] if existing else out

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("🔎 Search")
    query = st.text_input("Query", value="quantum cryptography robotics")
    max_results = st.slider("Max results", 50, 1000, 300, 50)

    st.header("🧠 NLP (Gemini)")
    do_summ = st.checkbox("Generate summaries", value=True)
    do_keys = st.checkbox("Extract keywords", value=False)
    max_words = st.slider("Summary length (words)", 30, 160, 60, 5)

    st.header("⚡ Performance")
    preview_n = st.slider("Preview count (process this many now)", 10, 100, 20)
    process_all = st.toggle("Process NLP for ALL results (slower)", value=False)

    go = st.button("Search")

# =========================
# Main logic
# =========================
if go:
    try:
        # ---- Fetch ----
        with st.spinner("Fetching papers..."):
            df = cached_fetch(query, max_results)
            if df is None or len(df) == 0:
                st.warning("No results found.")
                st.stop()

        # ---- Clean & deduplicate BEFORE NLP ----
        with st.spinner("Cleaning & deduplicating..."):
            df = remove_duplicates(df).reset_index(drop=True)
            df = ensure_columns(df)

            if "abstract" not in df.columns:
                df["abstract"] = ""

            # Clean abstract HTML → text
            df["abstract"] = df["abstract"].fillna("").astype(str).apply(cached_clean_html)

            # Normalize authors if present
            if "authors" in df.columns:
                df["authors"] = safe_normalize_authors(df["authors"])

        # ---- Select target rows for NLP (preview vs all) ----
        idx = df.index if process_all else df.index[:min(preview_n, len(df))]

        # ---- Summaries (Gemini) ----
        if do_summ:
            with st.spinner(f"Summarizing {len(idx)} abstracts with Gemini..."):
                df = run_summaries(df, idx, text_column="abstract", max_words=max_words)

        # ---- Keywords (Gemini) ----
        if do_keys:
            with st.spinner(f"Extracting keywords for {len(idx)} abstracts with Gemini..."):
                df = extract_keywords(df, text_column="abstract", top_n=5, idx=idx)

        # ---- Export-ready view ----
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
                "- **summary** — Gemini summary (if enabled)\n"
                "- **keywords** — Gemini keyphrases (if enabled)\n"
                "- **journal** — host venue/journal (if available)\n"
            )

        if (not process_all) and (do_summ or do_keys) and len(df) > len(idx):
            st.info("Want full coverage? Re-run with “Process NLP for ALL results”.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.info("Enter a query and click **Search**. For speed, NLP runs on a small preview first.")
