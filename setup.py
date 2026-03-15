"""
setup.py — one-time setup for the Bible Assistant project.

Run this before launching the app:
    python3 setup.py

What it does:
  1. Installs all required Python packages
  2. Checks that Ollama is running
  3. Pulls the required Ollama models (llama3 + nomic-embed-text)
  4. Cleans the raw KJV Bible text into bible_kjv_clean.txt
  5. Builds the Chroma vector database from the cleaned text
"""

import os
import subprocess
import sys


# ── 1. Install Python dependencies ───────────────────────────────────────────

PACKAGES = [
    "langchain",
    "langchain-ollama",
    "langchain-chroma",
    "langchain-community",
    "langchain-text-splitters",
    "streamlit",
]

print("==> Installing Python packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + PACKAGES)
print("    Done.\n")


# ── 2. Check Ollama is running ────────────────────────────────────────────────

import urllib.request, urllib.error  # noqa: E402  (import after install)

print("==> Checking Ollama...")
try:
    urllib.request.urlopen("http://localhost:11434", timeout=3)
    print("    Ollama is running.\n")
except urllib.error.URLError:
    print(
        "ERROR: Ollama does not appear to be running.\n"
        "       Start it with:  ollama serve\n"
        "       Then re-run this script."
    )
    sys.exit(1)


# ── 3. Pull required Ollama models ───────────────────────────────────────────

MODELS = ["llama3", "nomic-embed-text"]

for model in MODELS:
    print(f"==> Pulling Ollama model: {model}  (skipped if already present)...")
    subprocess.run(["ollama", "pull", model], check=True)
    print()


# ── 4. Clean the raw Bible text ──────────────────────────────────────────────

CLEAN_FILE = "bible_kjv_clean.txt"
RAW_FILE   = "bible_text_kjv.txt"

if os.path.exists(CLEAN_FILE):
    print(f"==> {CLEAN_FILE} already exists — skipping cleaning step.\n")
else:
    if not os.path.exists(RAW_FILE):
        print(
            f"ERROR: {RAW_FILE} not found.\n"
            "       Download the KJV Bible from gutenberg.org and save it as bible_text_kjv.txt"
        )
        sys.exit(1)
    print(f"==> Cleaning {RAW_FILE} → {CLEAN_FILE} ...")
    import clean_kjv  # noqa: E402
    clean_kjv.main()
    print()


# ── 5. Build the Chroma vector database ──────────────────────────────────────

DB_PATH = "./bible_db"

if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print(f"==> Vector database already exists at {DB_PATH} — skipping.\n")
else:
    print("==> Building vector database (this takes a few minutes)...")

    from langchain_ollama import OllamaEmbeddings                          # noqa: E402
    from langchain_chroma import Chroma                                    # noqa: E402
    from langchain_community.document_loaders import TextLoader            # noqa: E402
    from langchain_text_splitters import RecursiveCharacterTextSplitter    # noqa: E402

    loader = TextLoader(CLEAN_FILE)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits   = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=DB_PATH,
    )
    print(f"    Indexed {len(splits)} chunks into {DB_PATH}.\n")


# ── Done ──────────────────────────────────────────────────────────────────────

print("=" * 50)
print("Setup complete! Launch the app with:")
print()
print("    python3 -m streamlit run app.py")
print("=" * 50)
