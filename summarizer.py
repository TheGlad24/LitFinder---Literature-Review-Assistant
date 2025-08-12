# summarizer.py — Gemini API summarizer (fast)
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from .env

@st.cache_resource(show_spinner=False)
def _gemini(model_name: str = "gemini-1.5-flash"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Create a .env with GOOGLE_API_KEY=...")
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
        f"Summarize the following abstract in <= {max_words} words, preserve the main contribution and context.\n\n"
        f"ABSTRACT:\n{text.strip()}\n\n"
        "Return only the summary as plain text."
    )
    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"[Summarization error: {e}]"

def run_summaries(df, idx, text_column: str = "abstract", max_words: int = 60):
    """
    In-place: writes df.loc[idx, 'summary'] using Gemini.
    """
    for i in idx:
        df.at[i, "summary"] = summarize_abstract(str(df.at[i, text_column] or ""), max_words=max_words)
    return df
