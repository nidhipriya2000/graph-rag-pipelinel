"""
main.py
-------
CLI entry point for the Graph RAG pipeline.

Usage:
    # Interactive Q&A session
    python main.py

    # Single question
    python main.py --question "How does BERT use attention?"

    # Force rebuild (re-extract graph, re-embed documents)
    python main.py --rebuild

    # Show pipeline info
    python main.py --info

    # Visualize the knowledge graph
    python main.py --visualize
"""

import argparse
import sys

from src.pipeline import GraphRAGPipeline


CORPUS_PATH      = "data/corpus.txt"
GRAPH_CACHE_PATH = "data/knowledge_graph.json"
CHROMA_DB_PATH   = "data/chroma_db"

BANNER = """
╔══════════════════════════════════════════════════════╗
║          Graph RAG Pipeline  (Ollama + NetworkX)     ║
║          Type 'exit' or 'quit' to stop               ║
╚══════════════════════════════════════════════════════╝
"""


def build_pipeline(rebuild: bool = False) -> GraphRAGPipeline:
    pipeline = GraphRAGPipeline(
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        persist_directory=CHROMA_DB_PATH,
        graph_cache_path=GRAPH_CACHE_PATH,
        retrieval_k=5,
    )
    pipeline.build(
        CORPUS_PATH,
        force_rebuild_graph=rebuild,
        force_rebuild_vectors=rebuild,
    )
    return pipeline


def interactive_session(pipeline: GraphRAGPipeline) -> None:
    print(BANNER)
    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        result = pipeline.ask(question)

        print("\n" + "─" * 60)
        print(f"Answer:\n{result['answer']}")
        print("\nSources:", ", ".join(
            {d.metadata.get("source", "?") for d in result["source_docs"]}
        ))
        print("─" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph RAG Pipeline CLI")
    parser.add_argument("--question", "-q", type=str, help="Single question to answer")
    parser.add_argument("--rebuild",  "-r", action="store_true", help="Force rebuild pipeline")
    parser.add_argument("--info",           action="store_true", help="Print pipeline info")
    parser.add_argument("--visualize",      action="store_true", help="Save graph visualization")
    args = parser.parse_args()

    pipeline = build_pipeline(rebuild=args.rebuild)

    if args.info:
        import json
        print(json.dumps(pipeline.info(), indent=2, default=str))
        return

    if args.visualize:
        pipeline.visualize_graph("data/knowledge_graph.png")
        print("Graph saved to data/knowledge_graph.png")
        return

    if args.question:
        result = pipeline.ask(args.question)
        print(f"\nQ: {result['question']}")
        print(f"\nA: {result['answer']}")
        sources = {d.metadata.get("source", "?") for d in result["source_docs"]}
        print(f"\nSources: {', '.join(sources)}")
        return

    # Default: interactive session
    interactive_session(pipeline)


if __name__ == "__main__":
    main()
