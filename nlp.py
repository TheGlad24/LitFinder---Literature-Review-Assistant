# nlp.py — OpenAI GPT-4o mini keyword extraction (batched, robust, safe assignment)
import os
import json
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"

@st.cache_resource(show_spinner=False)
def _openai_client() -> OpenAI:
    api_key = ""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing. In Streamlit Cloud, set it under Settings → Secrets.")
    return OpenAI(api_key=api_key)

def _keywords_batch(texts: List[Tuple[int, str]], top_n: int = 5) -> Dict[int, List[str]]:
    """
    Extract keywords for a batch of (id, text). Returns {id: [keywords]} using one GPT call.
    Falls back per-item if JSON parsing fails.
    """
    client = _openai_client()
    items = [{"id": int(i), "text": (t or "")[:1200]} for i, t in texts]

    system = (
        "You are a scholarly assistant. Extract concise, domain-relevant keyphrases (1–3 words) "
        "that best describe each passage's topic and contribution. Avoid overly generic terms like 'paper', 'study', 'results'."
    )
    user = (
        "For each item, return exactly {k} keyphrases.\n"
        "STRICT JSON only: an array of objects with keys 'id' (int) and 'keywords' (array of strings).\n"
        "No code fences, no extra text.\n\n"
        "DATA:\n{data}"
    ).format(k=top_n, data=json.dumps(items, ensure_ascii=False))

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
        out: Dict[int, List[str]] = {}
        if isinstance(parsed, list):
            for obj in parsed:
                try:
                    kid = int(obj["id"])
                    kws = obj.get("keywords", [])
                    if isinstance(kws, list):
                        out[kid] = [str(x).strip() for x in kws if str(x).strip()]
                except Exception:
                    continue
        missing = {i for i, _ in texts} - set(out.keys())
        if missing:
            for mid in missing:
                text = next(t for i, t in texts if i == mid)
                out[mid] = _keywords_one(text, top_n=top_n)
        return out
    except Exception:
        return {i: _keywords_one(t, top_n=top_n) for i, t in texts}

def _keywords_one(text: str, top_n: int = 5) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    client = _openai_client()
    prompt = (
        f"Extract the {top_n} most salient, domain-relevant keyphrases (1–3 words) from the passage. "
        "Avoid generic words ('paper', 'study', 'results'). "
        "Return a strict JSON array of strings, no extra text.\n\n"
        f"PASSAGE:\n{text.strip()[:1200]}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip().strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip(": \n`")
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return []

def _chunks(seq: List[Tuple[int, str]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def extract_keywords(
    df: pd.DataFrame,
    text_column: str = "abstract",
    top_n: int = 5,
    idx: Optional[Iterable[int]] = None,
    batch_size: int = 12,
) -> pd.DataFrame:
    """
    Writes df.loc[idx, 'keywords'] using batched GPT-4o mini calls.
    Uses aligned Series assignment to avoid shape mismatch errors.
    """
    if idx is None:
        idx = df.index
    idx = pd.Index(idx)
    texts = df.loc[idx, text_column].fillna("").astype(str).tolist()
    id_text_pairs = list(zip(idx.tolist(), texts))

    results: Dict[int, List[str]] = {}
    for batch in _chunks(id_text_pairs, batch_size):
        results.update(_keywords_batch(batch, top_n=top_n))

    kw_lists = [results.get(i, []) for i in idx]
    df.loc[idx, "keywords"] = pd.Series(kw_lists, index=idx, dtype="object")
    return df
