"""
Benchmark retrieval latency on a loaded FAISS index.
Usage: python scripts/benchmark.py --n 50
"""

import argparse
import statistics
import time

from src.rag_pipeline import load_index, build_qa_chain, query

SAMPLE_QUESTIONS = [
    "What is the main topic of the document?",
    "Summarize the key findings.",
    "What methodology was used?",
    "List the main conclusions.",
    "What are the limitations mentioned?",
]


def run_benchmark(n: int = 50, embedding_provider: str = "openai", model: str = "gpt-4o-mini"):
    print(f"Loading index (embedding={embedding_provider}) …")
    vs = load_index(embedding_provider)
    chain = build_qa_chain(vs, model)

    latencies = []
    questions = (SAMPLE_QUESTIONS * ((n // len(SAMPLE_QUESTIONS)) + 1))[:n]

    print(f"Running {n} queries …\n")
    for i, q in enumerate(questions, 1):
        result = query(chain, q)
        latencies.append(result["latency_ms"])
        print(f"  [{i:3d}/{n}] {result['latency_ms']:5d}ms — {q[:50]}")

    print(f"\n{'─'*40}")
    print(f"  Queries  : {n}")
    print(f"  Mean     : {statistics.mean(latencies):.0f}ms")
    print(f"  Median   : {statistics.median(latencies):.0f}ms")
    print(f"  P95      : {sorted(latencies)[int(0.95 * n)]:.0f}ms")
    print(f"  Min/Max  : {min(latencies)}ms / {max(latencies)}ms")
    print(f"{'─'*40}")
    print(f"  Sub-1s   : {sum(1 for l in latencies if l < 1000)}/{n} queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--embeddings", default="openai", choices=["openai", "huggingface"])
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    run_benchmark(args.n, args.embeddings, args.model)
