# summarizer.py
import streamlit as st

@st.cache_resource(show_spinner=False)
def _load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

def summarize_abstract(abstract: str) -> str:
    if not abstract or not isinstance(abstract, str) or not abstract.strip():
        return "No abstract provided."
    text = abstract.strip()
    if len(text) > 1200:
        text = text[:1200]
    try:
        summarizer = _load_summarizer()
        out = summarizer(text, max_length=60, min_length=15, do_sample=False, truncation=True)[0]["summary_text"]
        return out.strip()
    except Exception as e:
        return f"Error during summarization: {e}"
