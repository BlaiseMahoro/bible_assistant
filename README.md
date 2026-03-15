# Bible Assistant

An AI-powered Bible study app that answers your questions using the full KJV Bible. Ask anything, get instant answers with exact verse citations — powered by LLaMA 3 and RAG. Built by BM.

---

## Use Case

Bible Assistant is built for anyone who wants to search and explore scripture using plain English questions instead of manually flipping through 31,000 verses.

**Who it's for:**
- Pastors and preachers researching topics for sermons
- Students studying scripture or theology
- Anyone curious about what the Bible says on a specific topic

**Example questions you can ask:**
- *"What does the Bible say about forgiveness?"*
- *"Which verses mention Jerusalem?"*
- *"What did Jesus say about prayer?"*
- *"What does Proverbs say about wisdom?"*

Every answer is grounded strictly in the KJV Bible text — no hallucinated verses, no outside knowledge. Each response includes exact book, chapter, and verse references.

---

## How It Works

The app uses **RAG (Retrieval-Augmented Generation)**:

1. The full KJV Bible (30,682 verses) is cleaned and stored in a local vector database
2. When you ask a question, the most relevant verses are retrieved
3. Those verses are passed to LLaMA 3, which generates an answer based only on that text
4. The answer includes citations and the source passages are shown in the UI

---

## Requirements

Before running the app, make sure you have the following installed:

- **Python 3.9+**
- **Ollama** — download from [ollama.com](https://ollama.com)

---

## Setup

### 1. Clone the project

```bash
git clone <your-repo-url>
cd bible_project
```

### 2. Add the raw Bible text

Download the KJV Bible plain text file from [gutenberg.org](https://gutenberg.org) and save it in the project folder as:

```
bible_text_kjv.txt
```

### 3. Start Ollama

In a separate terminal, run:

```bash
ollama serve
```

### 4. Run the setup script

This installs dependencies, pulls the required AI models, cleans the Bible text, and builds the vector database.

```bash
python3 setup.py
```

This only needs to be run once. It will:
- Install all Python packages
- Pull `llama3` and `nomic-embed-text` via Ollama
- Clean `bible_text_kjv.txt` → `bible_kjv_clean.txt`
- Build the `bible_db/` vector database

---

## Running the App

```bash
python3 -m streamlit run app.py
```

Then open your browser to:

```
http://localhost:8501
```

---

## Project Structure

```
bible_project/
├── app.py                  # Streamlit UI
├── my-bible-model.py       # Core RAG script (CLI version)
├── setup.py                # One-time setup script
├── clean_kjv.py            # Cleans raw KJV text into structured format
├── bible_text_kjv.txt      # Raw KJV Bible (Project Gutenberg)
├── bible_kjv_clean.txt     # Cleaned Bible — one verse per line
└── bible_db/               # Chroma vector database
```

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | LLaMA 3 via Ollama |
| Embeddings | Nomic Embed Text via Ollama |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| Bible Source | KJV — Project Gutenberg |

---

## Built by BM
