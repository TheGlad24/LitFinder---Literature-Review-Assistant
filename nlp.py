# nlp.py — Gemini Pro keyword extraction (robust JSON+fallback, Streamlit-cached)
import os
import json
from typing import Iterable, List, Dict, Optional

import pandas as pd
import streamlit as st
import google.generativeai as genai

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

def _parse_keywords(raw_text: str, top_n: int) -> List[str]:
    """
    Try JSON first; if that fails, fall back to splitting by commas/newlines.
    """
    s = (raw_text or "").strip().strip("` \n")
    # If model prefixed with 'json:' remove it
    if s.lower().startswith("json"):
        s = s[4:].lstrip(": \n`")
    # Try JSON array
    try:
        data = json.loads(s)
        if isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip()]
            return out[:top_n]
    except Exception:
        pass
    # Fallback: split heuristics
    parts = [p.strip() for p in s.replace("\n", ",").split(",") if p.strip()]
    return parts[:top_n]

def _keywords_one(text: str, top_n: int = 5) -> List[str]:
    """
    Extract concise, domain-relevant 1–3 word keyphrases.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    prompt = (
        f"Extract exactly {top_n} concise, domain-relevant keyphrases (each 1–3 words) "
        f"that best describe the topic and contribution of the passage. "
        f"Avoid generic words like 'paper', 'study', 'results'. "
        f"Return a strict JSON array of strings (no extra text).\n\n"
        f"PASSAGE:\n{text.strip()[:1200]}"
    )
    try:
        resp = _model.generate_content(prompt)
        return _parse_keywords(resp.text or "", top_n=top_n)
    except Exception:
        return []

def extract_keywords(
    df: pd.DataFrame,
    text_column: str = "abstract",
    top_n: int = 5,
    idx: Optional[Iterable[int]] = None,
    batch_size: int = 12,  # kept for API compatibility; per-item calls are more reliable here
) -> pd.DataFrame:
    """
    Writes df.loc[idx, 'keywords'] using Gemini Pro.
    """
    if idx is None:
        idx = df.index
    idx = pd.Index(idx)
    texts: List[str] = df.loc[idx, text_column].fillna("").astype(str).tolist()
    kw_lists = [_keywords_one(t, top_n=top_n) for t in texts]
    df.loc[idx, "keywords"] = pd.Series(kw_lists, index=idx, dtype="object")
    return df
