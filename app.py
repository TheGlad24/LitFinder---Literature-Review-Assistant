# app.py
import streamlit as st
import pandas as pd

from fetchers import fetch_all
from summarizer import summarize_abstract
from nlp import extract_keywords
from cleaning import clean_html_abstract, normalize_authors, remove_duplicates

# --- Streamlit page setup ---
st.set_page_config(page_title="LitFinder — Literature Review Assistant", layout="wide")
st.title("📚 LitFinder: Literature Review Assistant")
st.caption("Fast literature search with optional summarization & keywords. NLP runs on a preview first for speed.")

# --- Caching: network + lightweight transforms ---
@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(query: str, max_results: int):
    """Fetch results from APIs (OpenAlex/others via fetch_all) and return a DataFrame."""
    df = fetch_all(query, max_results=max_results, save_csv=False)
    return df

@st.cache_data(show_spinner=False)
def cached_clean_html(text: str) -> str:
    return clean_html_abstract(text)

# --- Helpers ---
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
    existing = [c for c in columns_to_keep if c in df.columns]

    out = df.copy()
    # Normalize authors column (list -> string)
    if "authors" in out.columns:
        out["authors"] = safe_normalize_authors(out["authors"])

    # Ensure keywords is a string list printable
    if "keywords" in out.columns:
        out["keywords"] = out["keywords"].apply(lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else (x or ""))

    # Clean HTML from abstracts if needed
    if "abstract" in out.columns:
        out["abstract"] = out["abstract"].fillna("").astype(str).apply(cached_clean_html)

    # Coerce year to int where possible
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    return out[existing] if existing else out

def run_summaries(df: pd.DataFrame, idx):
    texts = df.loc[idx, "abstract"].fillna("").astype(str).tolist()
    summaries = [summarize_abstract(t) for t in texts]
    df.loc[idx, "summary"] = summaries

# --- UI sidebar controls ---
with st.sidebar:
    st.header("🔎 Search")
    query = st.text_input("Query", value="quantum cryptography robotics")
    max_results = st.slider("Max results", min_value=50, max_value=1000, value=300, step=50)

    st.header("🧠 NLP Options")
    do_summ = st.checkbox("Generate summaries", value=True)
    do_keys = st.checkbox("Extract keywords (slower than summaries)", value=False)

    st.header("⚡ Performance")
    preview_n = st.slider("Preview count (how many items to process now)", 10, 100, 20)
    process_all = st.toggle("Process NLP for ALL results (can be slow)", value=False)

    go = st.button("Search")

# --- Main area logic ---
if go:
    try:
        with st.spinner("Fetching papers..."):
            df = cached_fetch(query, max_results)
            if df is None or len(df) == 0:
                st.warning("No results found.")
                st.stop()

        # Deduplicate BEFORE NLP for speed
        with st.spinner("Cleaning & deduplicating..."):
            df = remove_duplicates(df).reset_index(drop=True)
            df = ensure_columns(df)

            # Make sure we have 'abstract' column
            if "abstract" not in df.columns:
                df["abstract"] = ""

            # Clean abstracts (HTML → text)
            df["abstract"] = df["abstract"].fillna("").astype(str).apply(cached_clean_html)

            # Normalize authors if present
            if "authors" in df.columns:
                df["authors"] = safe_normalize_authors(df["authors"])

        # Decide how many rows to NLP
        if process_all:
            idx = df.index
        else:
            idx = df.index[:min(preview_n, len(df))]

        # Summaries
        if do_summ:
            with st.spinner(f"Summarizing {len(idx)} abstracts..."):
                run_summaries(df, idx)

        # Keywords
        if do_keys:
            with st.spinner(f"Extracting keywords for {len(idx)} abstracts..."):
                # your updated nlp.extract_keywords supports idx=...
                extract_keywords(df, text_column="abstract", top_n=5, idx=idx)

        # Prepare export view
        df_ready = clean_for_biblioshiny(df)

        # Feedback + preview
        st.success(f"✅ Done! {len(df)} papers processed. "
                   f"{'Full NLP ran.' if process_all else f'NLP preview on {len(idx)} rows.'}")

        st.subheader("🔍 Preview (first 20 rows)")
        st.dataframe(df_ready.head(20), use_container_width=True)

        st.download_button(
            "📥 Download CSV for Biblioshiny",
            df_ready.to_csv(index=False),
            file_name="litfinder_output.csv",
            mime="text/csv"
        )

        with st.expander("🧾 Columns explained"):
            st.markdown(
                "- **title** — paper title\n"
                "- **authors** — comma-separated list\n"
                "- **year** — publication year\n"
                "- **abstract** — cleaned abstract text\n"
                "- **summary** — 1–2 line abstractive summary (if enabled)\n"
                "- **keywords** — KeyBERT phrases (if enabled)\n"
                "- **journal** — host venue/journal (if available)\n"
            )

        # Optional: button to run NLP on the rest (if you started with preview)
        if (not process_all) and (do_summ or do_keys) and len(df) > len(idx):
            st.info("Want full coverage? Re-run with “Process NLP for ALL results”.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Enter a query and click **Search** to start. For speed, NLP runs on a small preview first.")
