"""
Gradio UI for RAG-Based Knowledge Assistant
Zero-setup access: python app.py  →  opens at http://localhost:7860
"""

import os
import time
from pathlib import Path

import gradio as gr

from src.rag_pipeline import (
    ingest_and_build,
    load_index,
    build_qa_chain,
    query,
    FAISS_INDEX_PATH,
)

# ── Global state ─────────────────────────────────────────
_chain = None
_vectorstore = None


def _get_embedding_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "openai")


def _get_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


# ── Gradio callbacks ──────────────────────────────────────

def ingest_files(files, progress=gr.Progress()):
    """Accept uploaded files, save to data/, rebuild index."""
    global _chain, _vectorstore

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    progress(0.1, desc="Saving uploaded files…")
    saved = []
    for f in files:
        dest = data_dir / Path(f.name).name
        dest.write_bytes(Path(f.name).read_bytes())
        saved.append(dest.name)

    progress(0.3, desc="Chunking & embedding… (this may take ~30s)")
    embedding_provider = _get_embedding_provider()
    _vectorstore = ingest_and_build(str(data_dir), embedding_provider)

    progress(0.9, desc="Building QA chain…")
    _chain = build_qa_chain(_vectorstore, _get_model())

    progress(1.0, desc="Done!")
    return f"✅ Indexed {len(saved)} file(s): {', '.join(saved)}"


def load_existing_index():
    """Load a pre-built FAISS index from disk."""
    global _chain, _vectorstore
    try:
        _vectorstore = load_index(_get_embedding_provider())
        _chain = build_qa_chain(_vectorstore, _get_model())
        return "✅ Loaded existing index."
    except FileNotFoundError:
        return "⚠️ No index found. Please upload documents first."


def answer_question(question: str, history: list):
    """Stream answer for a question."""
    global _chain

    if _chain is None:
        history.append((question, "⚠️ No index loaded. Upload documents or load existing index first."))
        return history

    result = query(_chain, question)
    answer = result["answer"]
    latency = result["latency_ms"]

    # Build source section
    source_lines = []
    for s in result["sources"]:
        source_lines.append(
            f"📄 **{Path(s['source']).name}**, p.{s['page']}\n> {s['snippet'][:150]}…"
        )
    sources_md = "\n\n".join(source_lines) if source_lines else "_No sources retrieved._"

    full_reply = f"{answer}\n\n---\n**Sources** *(retrieval: {latency}ms)*\n\n{sources_md}"
    history.append((question, full_reply))
    return history


# ── UI layout ─────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="RAG Knowledge Assistant",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=".source-box { font-size: 0.85em; color: #555; }",
    ) as demo:

        gr.Markdown(
            """
# 🔍 RAG-Based Knowledge Assistant
**Retrieval-Augmented Generation** over your PDFs and documents.  
Upload files → get cited answers in < 1 second on 10k-chunk corpora.
            """
        )

        with gr.Row():
            # ── Left panel: ingest ──
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Document Ingestion")

                upload = gr.File(
                    label="Upload PDFs / TXT / MD",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"],
                )
                ingest_btn = gr.Button("🔄 Ingest & Build Index", variant="primary")
                load_btn = gr.Button("📂 Load Existing Index")
                status = gr.Textbox(label="Status", interactive=False, lines=2)

                gr.Markdown(
                    """
---
**Embedding provider** (set via env var):  
`EMBEDDING_PROVIDER=openai` (default) or `huggingface`

**LLM model** (set via env var):  
`LLM_MODEL=gpt-4o-mini` (default)
                    """
                )

            # ── Right panel: chat ──
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions")
                chatbot = gr.Chatbot(height=480, show_copy_button=True)
                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="Ask anything about your documents…",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ Clear chat", size="sm")

        # ── Wire up ──
        ingest_btn.click(ingest_files, inputs=[upload], outputs=[status])
        load_btn.click(load_existing_index, outputs=[status])
        send_btn.click(answer_question, inputs=[question_box, chatbot], outputs=[chatbot])
        question_box.submit(answer_question, inputs=[question_box, chatbot], outputs=[chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False, server_name="0.0.0.0", server_port=7860)
