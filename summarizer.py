# summarizer.py — Gemini API summarizer using Streamlit secrets
import os
import streamlit as st
import google.generativeai as genai
from typing import Iterable

@st.cache_resource(show_spinner=False)
def _gemini(model_name: str = "gemini-1.5-flash"):
    """
    Loads a Gemini model once per session.
    API key is read from Streamlit secrets: st.secrets["GOOGLE_API_KEY"].
    Falls back to env var GOOGLE_API_KEY if secrets are not set.
    """
    api_key = ""
    # 1) Streamlit Cloud secret
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    # 2) Local fallback (optional)
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. "
            "On Streamlit Cloud, set it under Settings → Secrets with key 'GOOGLE_API_KEY'."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def summarize_abstract(text: str, max_words: int = 60) -> str:
    """
    Summarize an academic abstract in <= max_words, preserving the main contribution.
    """
    if not isinstance(text, str) or not text.strip():
        return "No abstract provided."
    model = _gemini()
    prompt = (
        "You are a scholarly abstract summarizer.\n"
        f"Summarize the following abstract in <= {max_words} words, "
        "preserving the main contribution and context.\n\n"
        f"ABSTRACT:\n{text.strip()}\n\n"
        "Return only the summary as plain text."
    )
    try:
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"[Summarization error: {e}]"

def run_summaries(df, idx: Iterable[int], text_column: str = "abstract", max_words: int = 60):
    """
    In-place: writes df.loc[idx, 'summary'] using Gemini.
    """
    for i in idx:
        df.at[i, "summary"] = summarize_abstract(str(df.at[i, text_column] or ""), max_words=max_words)
    return df
