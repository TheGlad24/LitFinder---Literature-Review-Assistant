# LitFinder — Literature Review Assistant 📚🔍

LitFinder is a **command-line based toolkit** for accelerating the literature review process. It helps researchers quickly **fetch, clean, summarize, and analyze** academic papers using NLP techniques. Built with Python, it's designed for speed, simplicity, and clarity — making it an ideal assistant for both students and professionals.

---

## 🚀 Features

- 🔎 **Paper Fetching**: Pull metadata and abstracts directly from academic databases (e.g., arXiv).
- 🧹 **Data Cleaning**: Strip HTML tags, normalize whitespace, and prepare abstracts for NLP.
- 🧠 **NLP Summarization**: Generate short, high-quality summaries of long abstracts using transformers.
- ⚙️ **Modular Design**: Separate files for fetching, cleaning, summarizing, and utility functions.

---

## 🗂️ Folder Structure

Litfinder/
├── app.py # Entry point for running summarization
├── summarizer.py # Contains the core summarization logic
├── cleaning.py # Preprocessing and abstract cleaning
├── fetchers.py # Handles fetching abstracts and metadata
├── nlp.py # NLP models and transformer wrappers
├── utils.py # Utility functions (e.g., token count)
├── README.md # You’re reading it!

yaml
Copy
Edit

---

## 🛠️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/TheGlad24/LitFinder---Literature-Review-Assistant.git
cd LitFinder---Literature-Review-Assistant
2. Create a virtual environment
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Run the summarizer
bash
Copy
Edit
python app.py
You will be prompted to enter a paper title or query. The system fetches the abstract, cleans it, and provides a concise summary.

📋 Example Output
vbnet
Copy
Edit
Enter your paper title or keyword: Quantum Computing and Robotics
Fetching paper...
Cleaning text...
Generating summary...

Summary:
"Quantum computing is emerging as a powerful tool for enhancing optimization and learning in robotics, particularly in areas like path planning and cryptographic security."
📦 Dependencies
transformers by HuggingFace

requests

beautifulsoup4

tqdm

nltk

🤖 NLP Approach
The summarizer uses a pretrained transformer model (e.g., t5-small) to perform abstractive summarization. The pipeline includes:

Cleaning text using BeautifulSoup

Reducing token count for model input limits

Generating concise summaries via HuggingFace pipeline("summarization")

🌍 Why LitFinder?
Writing literature reviews is time-consuming. LitFinder helps by:

Reducing overhead from reading dozens of full papers

Generating focused summaries instantly

Supporting batch operations for automated reviews

📜 License
This project is open-sourced under the MIT License.

🙋‍♂️ Contributions
Pull requests are welcome! If you find bugs, want to improve the model, or suggest enhancements, feel free to open an issue or fork the repo.

🧠 Future Additions
GUI version using Streamlit

PDF upload and summarization

Semantic similarity scoring across papers

BibTeX + citation integration

