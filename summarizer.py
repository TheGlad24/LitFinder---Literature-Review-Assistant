# summarizer.py — Gemini Pro summarizer (robust, Streamlit-cached)
import os
from typing import Iterable, List

import pandas as pd
import streamlit as st
import google.generativeai as genai

# ---------- Setup ----------
API_KEY_ENV = "GOOGLE_API_KEY"
MODEL_NAME = "gemini-pro"

@st.cache_resource(show_spinner=False)
def _configure_gemini():
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise ValueError(
            f"{API_KEY_ENV} missing. Set an environment variable like:\n"
            f'  export {API_KEY_ENV}="your_api_key_here"\n'
            "If running on Streamlit Cloud, add it under Settings → Secrets."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)

_model = _configure_gemini()

# ---------- Core ----------
def _summarize_one(text: str, max_words: int = 60) -> str:
    """
    Summarize a single abstract. Keeps it concise and focused on contribution/method/context.
    """
    if not isinstance(text, str) or not text.strip():
        return "No abstract provided."
    prompt = (
        f"Summarize the following academic abstract in <= {max_words} words. "
        f"Preserve the main contribution, method, and domain context. "
        f"Return only the summary as plain text.\n\n"
        f"ABSTRACT:\n{text.strip()[:1200]}"
    )
    try:
        resp = _model.generate_content(prompt)
        out = (resp.text or "").strip()
        # ultra-short fallback if model returns something empty
        return out if out else "Summary unavailable."
    except Exception as e:
        return f"[Summarization error: {e}]"

def run_summaries(
    df: pd.DataFrame,
    idx: Iterable[int],
    text_column: str = "abstract",
    max_words: int = 60,
    batch_size: int = 12,  # kept for API compatibility; Gemini path summarizes per-item
) -> pd.DataFrame:
    """
    Writes df.loc[idx, 'summary'] using Gemini Pro.
    Per-item calls for robustness (Gemini may be less strict on JSON).
    """
    idx = pd.Index(idx)
    texts: List[str] = df.loc[idx, text_column].fillna("").astype(str).tolist()
    summaries = [_summarize_one(t, max_words=max_words) for t in texts]
    df.loc[idx, "summary"] = pd.Series(summaries, index=idx, dtype="object")
    return df
