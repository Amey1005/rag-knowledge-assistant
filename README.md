# 🔍 RAG-Based Knowledge Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-orange)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00)](https://gradio.app)

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that ingests PDFs and documents, chunks and indexes them in FAISS, then answers natural-language queries with **cited source snippets** — all in **< 1 second** on 10k-chunk corpora.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Multi-format ingestion** | PDFs, TXT, Markdown via LangChain loaders |
| 🧩 **Smart chunking** | `RecursiveCharacterTextSplitter` (512 tokens, 64 overlap) |
| ⚡ **Sub-second retrieval** | FAISS flat index with cosine similarity, top-5 chunks |
| 🔄 **Swappable embeddings** | OpenAI `text-embedding-3-small` **or** HuggingFace `all-MiniLM-L6-v2` — single config change |
| 📚 **Cited answers** | Every response includes `[Source: file, Page: n]` references |
| 🖥️ **Gradio UI** | Zero-setup web interface at `localhost:7860` |
| 🔧 **CLI mode** | Headless ingest + interactive REPL |

---

## 🏗️ Architecture

```
PDFs / Docs
    │
    ▼
┌───────────────────────────────┐
│  Document Loaders (LangChain) │  ← PyPDFLoader, TextLoader
└───────────────┬───────────────┘
                │
    ▼
┌───────────────────────────────┐
│  RecursiveCharacterTextSplitter│  chunk_size=512, overlap=64
└───────────────┬───────────────┘
                │ chunks
    ▼
┌───────────────────────────────┐
│  Embeddings                   │  OpenAI / HuggingFace (switchable)
└───────────────┬───────────────┘
                │ vectors
    ▼
┌───────────────────────────────┐
│  FAISS Index  (persisted)     │  saved to faiss_index/
└───────────────┬───────────────┘
                │ top-k retrieval
    ▼
┌───────────────────────────────┐
│  RetrievalQA Chain            │  GPT-4o-mini + citation prompt
└───────────────┬───────────────┘
                │
    ▼
  Answer + Source Citations
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Amey1005/rag-knowledge-assistant.git
cd rag-knowledge-assistant
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add documents

```bash
mkdir data
cp your_documents/*.pdf data/
```

### 4a. Launch Gradio UI (recommended)

```bash
python app.py
# → opens http://localhost:7860
```

Upload documents in the UI, click **Ingest & Build Index**, then ask questions.

### 4b. CLI mode

```bash
# Ingest + build index
python -m src.rag_pipeline --ingest --data-dir data

# Ask questions interactively
python -m src.rag_pipeline
```

---

## 🔄 Switching Embedding Providers

The pipeline supports two embedding backends with **zero code changes** — just set an environment variable:

```bash
# OpenAI (default, best quality)
EMBEDDING_PROVIDER=openai python app.py

# HuggingFace (fully offline, no API key needed)
EMBEDDING_PROVIDER=huggingface python app.py
```

| Provider | Model | Requires API key | Latency |
|---|---|---|---|
| `openai` | `text-embedding-3-small` | ✅ Yes | ~200ms |
| `huggingface` | `all-MiniLM-L6-v2` | ❌ No | ~50ms (CPU) |

> **Note:** You must rebuild the index when switching providers.

---

## 📊 Performance

Benchmarked on a 10,000-chunk corpus (mixed PDFs):

| Metric | Value |
|---|---|
| Mean retrieval latency | **< 1 second** |
| Index build time (10k chunks, OpenAI) | ~45s |
| Index build time (10k chunks, HuggingFace) | ~12s |
| Top-k retrieved | 5 |

Run your own benchmark:

```bash
python scripts/benchmark.py --n 100 --embeddings openai
```

---

## 📁 Project Structure

```
rag-knowledge-assistant/
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py      # Core: load → chunk → embed → query
├── scripts/
│   └── benchmark.py         # Latency benchmarking
├── app.py                   # Gradio web UI
├── data/                    # Your PDFs go here (git-ignored)
├── faiss_index/             # Persisted FAISS index (git-ignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ Tech Stack

- **LangChain** — document loaders, text splitting, retrieval chains
- **FAISS** — high-performance vector similarity search (Facebook AI)
- **OpenAI Embeddings** — `text-embedding-3-small` (768-dim)
- **HuggingFace** — `all-MiniLM-L6-v2` offline fallback
- **GPT-4o-mini** — answer generation with citation prompting
- **Gradio** — zero-config web UI

---

## 🧪 Example Output

```
You: What is the main argument of Chapter 3?

Assistant (847ms):
The main argument of Chapter 3 is that neural scaling laws follow a
predictable power-law relationship with compute budget...

[Source: attention_paper.pdf, Page: 12]
[Source: attention_paper.pdf, Page: 14]
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Amey Kushare](https://github.com/Amey1005)*
