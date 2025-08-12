# nlp.py — Gemini API keyword extractor using Streamlit secrets (no KeyBERT)
import os
import json
import streamlit as st
import google.generativeai as genai
import pandas as pd
from typing import Iterable, Optional

@st.cache_resource(show_spinner=False)
def _gemini(model_name: str = "gemini-1.5-flash"):
    """
    Loads a Gemini model once per session.
    API key is read from Streamlit secrets: st.secrets["GOOGLE_API_KEY"].
    Falls back to env var GOOGLE_API_KEY if secrets are not set.
    """
    api_key = ""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. "
            "On Streamlit Cloud, set it under Settings → Secrets with key 'GOOGLE_API_KEY'."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def _gemini_keywords(text: str, top_n: int = 5):
    """
    Ask Gemini for top-N domain-relevant keyphrases. Returns List[str].
    """
    if not isinstance(text, str) or not text.strip():
        return []
    model = _gemini()
    prompt = (
        "Extract the most salient, domain-relevant keyphrases from the passage.\n"
        f"Return a strict JSON array of {top_n} strings (no code fences, no explanations).\n\n"
        f"PASSAGE:\n{text.strip()}"
    )
    try:
        resp = model.generate_content(prompt)
        raw = (getattr(resp, "text", "") or "").strip()
        # Strip accidental backticks/labels
        raw = raw.strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip(" :\n`")
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if isinstance(x, (str, int, float))]
    except Exception:
        pass
    return []

def extract_keywords(
    df: pd.DataFrame,
    text_column: str = "abstract",
    top_n: int = 5,
    idx: Optional[Iterable[int]] = None
) -> pd.DataFrame:
    """
    In-place: writes df.loc[idx, 'keywords'] using Gemini.
    """
    if idx is None:
        idx = df.index
    out = []
    for i in idx:
        text = str(df.at[i, text_column] or "")
        out.append(_gemini_keywords(text, top_n=top_n))
    df.loc[idx, "keywords"] = out
    return df
