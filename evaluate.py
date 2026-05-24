"""
evaluate.py
-----------
Runs the RAGAS evaluation suite against the Graph RAG pipeline
and saves a JSON report to data/evaluation_report.json.

Usage:
    python evaluate.py
    python evaluate.py --rebuild      # force rebuild before evaluating
    python evaluate.py --no-save      # print results only, don't save
"""

import argparse
import json

from src.pipeline import GraphRAGPipeline
from src.evaluator import RAGASEvaluator


CORPUS_PATH = "data/corpus.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Graph RAG pipeline")
    parser.add_argument("--rebuild",  action="store_true", help="Force rebuild pipeline")
    parser.add_argument("--no-save",  action="store_true", help="Skip saving report to disk")
    args = parser.parse_args()

    # Build pipeline
    print("Initializing Graph RAG pipeline…")
    pipeline = GraphRAGPipeline(
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        retrieval_k=5,
    )
    pipeline.build(
        CORPUS_PATH,
        force_rebuild_graph=args.rebuild,
        force_rebuild_vectors=args.rebuild,
    )

    # Run RAGAS
    evaluator = RAGASEvaluator(pipeline, judge_model="llama3.2")
    results = evaluator.evaluate()

    # Print report
    evaluator.print_report(results)

    # Suggestions
    suggestions = RAGASEvaluator.suggest_improvements(results["scores"])
    print("Improvement suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")
    print()

    # Save report
    if not args.no_save:
        evaluator.save_report(results)
        print("Report saved to data/evaluation_report.json")


if __name__ == "__main__":
    main()
