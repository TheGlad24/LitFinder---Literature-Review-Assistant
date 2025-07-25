# summarizer.py
from transformers import pipeline

# Load the distilbart summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_abstract(abstract):
    """
    Summarize an academic abstract using DistilBART. 
    Returns a short one-sentence summary or a fallback message.
    """
    if not abstract or not isinstance(abstract, str) or abstract.strip() == "":
        return "No abstract provided."

    try:
        summary = summarizer(abstract, max_length=60, min_length=15, do_sample=False)[0]["summary_text"]
        return summary.strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"
