# nlp.py
import streamlit as st
import pandas as pd

@st.cache_resource(show_spinner=False)
def _kw_model():
    from keybert import KeyBERT
    return KeyBERT(model='all-MiniLM-L6-v2')

def extract_keywords(df: pd.DataFrame, text_column='abstract', top_n=5, idx=None):
    if idx is None:
        idx = df.index
    km = _kw_model()
    def _one(t):
        if not isinstance(t, str) or not t.strip(): return []
        kws = km.extract_keywords(
            t[:1000], top_n=top_n, keyphrase_ngram_range=(1,2),
            stop_words='english', use_mmr=True, diversity=0.7
        )
        return [k for k,_ in kws]
    df.loc[idx, "keywords"] = df.loc[idx, text_column].apply(_one)
    return df
