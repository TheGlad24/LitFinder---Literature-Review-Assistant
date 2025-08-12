# summarizer.py — OpenAI GPT-4o mini summarizer (batched, robust, safe assignment)
import os
import json
from typing import Iterable, List, Dict, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"

@st.cache_resource(show_spinner=False)
def _openai_client() -> OpenAI:
    """
    Load OpenAI client once. Reads key from Streamlit secrets (preferred) or env var.
    """
    api_key = ""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing. In Streamlit Cloud, set it under Settings → Secrets.")
    return OpenAI(api_key=api_key)

def _summarize_batch(texts: List[Tuple[int, str]], max_words: int = 60) -> Dict[int, str]:
    """
    Summarize a batch of (id, text). Returns {id: summary}. Uses a single GPT call.
    Falls back per-item if JSON parsing fails.
    """
    client = _openai_client()
    # Prepare compact payload; truncate very long abstracts for speed/cost.
    items = [{"id": int(i), "abstract": (t or "")[:1200]} for i, t in texts]

    system = (
        "You are a scholarly assistant. Summarize academic abstracts succinctly, "
        "preserving the main contribution, method, and domain context."
    )
    user = (
        "Summarize each abstract in <= {mw} words.\n"
        "Return STRICT JSON: an array of objects with keys 'id' (int) and 'summary' (string).\n"
        "No code fences, no extra text.\n\n"
        "DATA:\n{data}"
    ).format(mw=max_words, data=json.dumps(items, ensure_ascii=False))

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content.strip()
        content = content.strip("` \n")
        if content.lower().startswith("json"):
            content = content[4:].lstrip(": \n`")
        parsed = json.loads(content)
        out = {}
        if isinstance(parsed, list):
            for obj in parsed:
                try:
                    out[int(obj["id"])] = str(obj["summary"]).strip()
                except Exception:
                    continue
        # If model returned fewer than requested, fill missing with per-item fallback
        missing = {i for i, _ in texts} - set(out.keys())
        if missing:
            for mid in missing:
                abs_text = next(t for i, t in texts if i == mid)
                out[mid] = _summarize_one(abs_text, max_words=max_words)
        return out
    except Exception:
        # Fallback: per-item
        return {i: _summarize_one(t, max_words=max_words) for i, t in texts}

def _summarize_one(text: str, max_words: int = 60) -> str:
    if not isinstance(text, str) or not text.strip():
        return "No abstract provided."
    client = _openai_client()
    prompt = (
        f"Summarize the following academic abstract in <= {max_words} words, "
        "preserving main contribution, method, and domain context. "
        "Return only the summary as plain text.\n\n"
        f"ABSTRACT:\n{text.strip()[:1200]}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Summarization error: {e}]"

def _chunks(seq: List[Tuple[int, str]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def run_summaries(
    df: pd.DataFrame,
    idx: Iterable[int],
    text_column: str = "abstract",
    max_words: int = 60,
    batch_size: int = 12,
) -> pd.DataFrame:
    """
    Writes df.loc[idx, 'summary'] using batched GPT-4o mini calls.
    Uses aligned Series assignment to avoid shape mismatch errors.
    """
    idx = pd.Index(idx)
    texts = df.loc[idx, text_column].fillna("").astype(str).tolist()
    id_text_pairs = list(zip(idx.tolist(), texts))

    results: Dict[int, str] = {}
    for batch in _chunks(id_text_pairs, batch_size):
        results.update(_summarize_batch(batch, max_words=max_words))

    # Build aligned series
    summaries = [results.get(i, "") for i in idx]
    df.loc[idx, "summary"] = pd.Series(summaries, index=idx, dtype="object")
    return df
