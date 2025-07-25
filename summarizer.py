import streamlit as st
import openai

# Get API key from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def summarize_abstract(abstract):
    if not abstract or abstract.strip() == "":
        return "No abstract provided."

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes academic abstracts in one sentence."
                },
                {
                    "role": "user",
                    "content": f"Summarize this abstract in one sentence:\n\n{abstract}"
                }
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"
