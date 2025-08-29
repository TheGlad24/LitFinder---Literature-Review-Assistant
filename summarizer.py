# summarizer.py — Gemini Pro summarizer
import os
import pandas as pd
import streamlit as st
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-pro"
model = genai.GenerativeModel(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def _gemini_model():
    return model

def _summarize_one(text: str, max_words: int = 60) -> str:
    if not isinstance(text, str) or not text.strip():
        return "No abstract provided."
    prompt = f"Summarize the following academic abstract in <= {max_words} words:\n\n{text[:1200]}"
    try:
        response = _gemini_model().generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Summarization error: {e}]"

def run_summaries(
    df: pd.DataFrame,
    idx,
    text_column: str = "abstract",
    max_words: int = 60,
    batch_size: int = 12,
) -> pd.DataFrame:
    idx = pd.Index(idx)
    texts = df.loc[idx, text_column].fillna("").astype(str).tolist()
    summaries = [_summarize_one(t, max_words=max_words) for t in texts]
    df.loc[idx, "summary"] = pd.Series(summaries, index=idx, dtype="object")
    return df
