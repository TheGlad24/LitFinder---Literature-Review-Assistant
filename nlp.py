# nlp.py — Gemini Pro keyword extraction
import os, json
import pandas as pd
import streamlit as st
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-pro"
model = genai.GenerativeModel(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def _gemini_model():
    return model

def _keywords_one(text: str, top_n: int = 5) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    prompt = (
        f"Extract {top_n} concise, domain-relevant keyphrases (1–3 words) "
        f"from the following passage. Avoid generic words.\n\n{text[:1200]}"
    )
    try:
        response = _gemini_model().generate_content(prompt)
        # Gemini might output plain text → split by commas/newlines
        raw = response.text.strip()
        data = [kw.strip() for kw in raw.replace("\n", ",").split(",") if kw.strip()]
        return data[:top_n]
    except Exception:
        return []

def extract_keywords(
    df: pd.DataFrame,
    text_column: str = "abstract",
    top_n: int = 5,
    idx=None,
    batch_size: int = 12,
) -> pd.DataFrame:
    if idx is None:
        idx = df.index
    idx = pd.Index(idx)
    texts = df.loc[idx, text_column].fillna("").astype(str).tolist()
    kw_lists = [_keywords_one(t, top_n=top_n) for t in texts]
    df.loc[idx, "keywords"] = pd.Series(kw_lists, index=idx, dtype="object")
    return df
