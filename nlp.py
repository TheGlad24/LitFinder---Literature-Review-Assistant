from keybert import KeyBERT
import pandas as pd

# Load the model once globally
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_keywords(df, text_column='abstract', top_n=5):
    keywords_list = []

    for text in df[text_column].fillna(""):
        if not isinstance(text, str) or text.strip() == "":
            keywords_list.append([])
        else:
            try:
                keywords = kw_model.extract_keywords(
                    text,
                    top_n=top_n,
                    stop_words='english'
                )
                keywords_list.append([kw[0] for kw in keywords])
            except Exception:
                keywords_list.append([])

    df["keywords"] = keywords_list
    return df
