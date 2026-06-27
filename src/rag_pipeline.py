"""
RAG-Based Knowledge Assistant
Core pipeline: ingest PDFs/docs → chunk → embed → FAISS index → query with citations
"""

import os
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
FAISS_INDEX_PATH = "faiss_index"

CITATION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise knowledge assistant. Use ONLY the provided context to answer.
Always cite source snippets at the end of your answer in the format [Source: <filename>, Page: <page>].

Context:
{context}

Question: {question}

Answer (with citations):""",
)


# ─────────────────────────────────────────────────────────
# Embedding selector  (swap with single config change)
# ─────────────────────────────────────────────────────────

def get_embeddings(provider: str = "openai"):
    """
    provider: "openai" | "huggingface"
    Switch between providers with a single config change.
    """
    if provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# ─────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────

def load_documents(data_dir: str = "data") -> list[Document]:
    """Load PDFs and .txt files from data_dir."""
    docs = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Add PDFs/docs there.")

    # PDFs
    pdf_loader = DirectoryLoader(
        data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    docs.extend(pdf_loader.load())

    # Plain text / markdown
    for txt_file in data_path.rglob("*.txt"):
        docs.extend(TextLoader(str(txt_file)).load())
    for md_file in data_path.rglob("*.md"):
        docs.extend(TextLoader(str(md_file)).load())

    print(f"[Ingest] Loaded {len(docs)} raw document pages/chunks")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into fixed-size chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"[Chunk]  {len(docs)} pages → {len(chunks)} chunks  "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ─────────────────────────────────────────────────────────
# Index
# ─────────────────────────────────────────────────────────

def build_index(
    chunks: list[Document],
    embedding_provider: str = "openai",
    save_path: str = FAISS_INDEX_PATH,
) -> FAISS:
    """Embed chunks and persist FAISS index to disk."""
    embeddings = get_embeddings(embedding_provider)
    print(f"[Index]  Building FAISS index with {embedding_provider} embeddings …")
    t0 = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.perf_counter() - t0
    vectorstore.save_local(save_path)
    print(f"[Index]  Done in {elapsed:.1f}s — saved to '{save_path}/'")
    return vectorstore


def load_index(
    embedding_provider: str = "openai",
    load_path: str = FAISS_INDEX_PATH,
) -> FAISS:
    """Load a previously saved FAISS index."""
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"No index found at '{load_path}/'. Run ingest first."
        )
    embeddings = get_embeddings(embedding_provider)
    vectorstore = FAISS.load_local(
        load_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[Index]  Loaded index from '{load_path}/'")
    return vectorstore


# ─────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────

def build_qa_chain(vectorstore: FAISS, model: str = "gpt-4o-mini") -> RetrievalQA:
    """Wrap vectorstore + LLM in a RetrievalQA chain."""
    llm = ChatOpenAI(model=model, temperature=0)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CITATION_PROMPT},
    )
    return chain


def query(chain: RetrievalQA, question: str) -> dict:
    """Run a question through the chain; return answer + source snippets."""
    t0 = time.perf_counter()
    result = chain.invoke({"query": question})
    latency = time.perf_counter() - t0

    sources = []
    for doc in result.get("source_documents", []):
        meta = doc.metadata
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "snippet": doc.page_content[:200].replace("\n", " "),
        })

    return {
        "answer": result["result"],
        "latency_ms": round(latency * 1000),
        "sources": sources,
    }


# ─────────────────────────────────────────────────────────
# CLI helper
# ─────────────────────────────────────────────────────────

def ingest_and_build(data_dir: str = "data", embedding_provider: str = "openai"):
    """Full ingest → chunk → index pipeline."""
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    vectorstore = build_index(chunks, embedding_provider)
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Knowledge Assistant CLI")
    parser.add_argument("--ingest", action="store_true", help="Re-ingest and rebuild index")
    parser.add_argument("--data-dir", default="data", help="Directory with PDFs/docs")
    parser.add_argument("--embeddings", default="openai", choices=["openai", "huggingface"])
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if args.ingest:
        vs = ingest_and_build(args.data_dir, args.embeddings)
    else:
        vs = load_index(args.embeddings)

    chain = build_qa_chain(vs, args.model)
    print("\nRAG Knowledge Assistant ready. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = query(chain, q)
        print(f"\nAssistant ({out['latency_ms']}ms):\n{out['answer']}")
        print("\nSources:")
        for s in out["sources"]:
            print(f"  [{s['source']}, p.{s['page']}] {s['snippet'][:100]}…")
        print()
