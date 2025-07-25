# summarizer.py
from transformers import pipeline
import torch
import pandas as pd

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def summarize_abstracts(df, text_column="abstract", output_column="summary", batch_size=8):
    summaries = []
    texts = df[text_column].fillna("").tolist()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        clean_batch = [text if text.strip() else "No abstract provided." for text in batch]

        try:
            outputs = summarizer(
                clean_batch,
                max_length=100,
                min_length=20,
                do_sample=False,
                truncation=True
            )
            batch_summaries = [out['summary_text'] for out in outputs]
        except Exception as e:
            batch_summaries = [f"Error: {str(e)}"] * len(clean_batch)

        summaries.extend(batch_summaries)

    df[output_column] = summaries
    return df
