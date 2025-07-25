# summarizer.py
from transformers import pipeline
import torch

# Use GPU if available for faster performance
device = 0 if torch.cuda.is_available() else -1

# Load summarization pipeline with BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def summarize_abstract(abstract):
    if not abstract or abstract.strip() == "":
        return "No abstract provided."

    try:
        # HuggingFace models have a max token limit (e.g. ~1024 for BART)
        summary = summarizer(abstract, max_length=100, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"
