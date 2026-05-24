"""
pipeline.py
-----------
Orchestrates the Graph RAG pipeline end-to-end:
  1. Load corpus → build knowledge graph → build vector store
  2. Initialise HybridRetriever
  3. Construct a LangChain RetrievalQA-style chain with graph-augmented context
  4. Expose a clean ask() interface for Q&A
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.graph_builder import KnowledgeGraphBuilder
from src.vector_store import VectorStoreManager
from src.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
QA_PROMPT_TEMPLATE = """You are an expert research assistant with access to a knowledge graph
and a set of relevant documents. Use the context below to answer the question accurately.

--- KNOWLEDGE GRAPH CONTEXT ---
{graph_context}

--- RETRIEVED DOCUMENT CONTEXT ---
{document_context}

--- QUESTION ---
{question}

Instructions:
- Base your answer primarily on the provided context.
- If the context does not contain enough information, say so clearly.
- Be concise but thorough. Cite key facts from the context.
- Do NOT hallucinate entities, dates, or model names.

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["graph_context", "document_context", "question"],
    template=QA_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# GraphRAGPipeline
# ---------------------------------------------------------------------------
class GraphRAGPipeline:
    """
    Full Graph RAG pipeline.

    Quick start:
        pipeline = GraphRAGPipeline()
        pipeline.build("data/corpus.txt")
        answer = pipeline.ask("Who developed BERT and what corpus was it trained on?")
        print(answer["answer"])
    """

    def __init__(
        self,
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        persist_directory: str = "data/chroma_db",
        graph_cache_path: str = "data/knowledge_graph.json",
        retrieval_k: int = 5,
        graph_hop_depth: int = 1,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.persist_directory = persist_directory
        self.graph_cache_path = graph_cache_path
        self.retrieval_k = retrieval_k

        # Core components
        self.graph_builder = KnowledgeGraphBuilder(
            model=llm_model, base_url=base_url
        )
        self.vector_store_manager = VectorStoreManager(
            embedding_model=embedding_model,
            base_url=base_url,
            persist_directory=persist_directory,
        )
        self.retriever: Optional[HybridRetriever] = None
        self.llm = OllamaLLM(model=llm_model, base_url=base_url, temperature=0.1)

        self._is_built = False

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    def build(
        self,
        corpus_path: str,
        force_rebuild_graph: bool = False,
        force_rebuild_vectors: bool = False,
    ) -> None:
        """
        Loads the corpus and constructs (or loads cached) knowledge graph
        and vector store, then wires up the hybrid retriever.
        """
        # 1. Load corpus
        logger.info("Loading corpus from %s", corpus_path)
        documents = self.graph_builder.load_corpus(corpus_path)

        # 2. Knowledge graph
        graph_cache = Path(self.graph_cache_path)
        if graph_cache.exists() and not force_rebuild_graph:
            logger.info("Loading cached knowledge graph from %s", self.graph_cache_path)
            self.graph_builder.load_graph(self.graph_cache_path)
        else:
            logger.info("Building knowledge graph (this may take a few minutes)…")
            self.graph_builder.build_graph(documents)
            self.graph_builder.save_graph(self.graph_cache_path)
            logger.info("Knowledge graph saved to %s", self.graph_cache_path)

        # 3. Vector store
        self.vector_store_manager.build(
            documents, force_rebuild=force_rebuild_vectors
        )

        # 4. Hybrid retriever
        self.retriever = HybridRetriever(
            graph_builder=self.graph_builder,
            vector_store=self.vector_store_manager,
            graph_hop_depth=1,
        )

        self._is_built = True
        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Ask / Q&A
    # ------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        """
        Runs the full Graph RAG pipeline for a single question.

        Returns a dict with:
            answer          – the LLM's response string
            question        – the original question
            source_docs     – list of retrieved Document objects
            graph_context   – the graph context injected into the prompt
        """
        self._check_built()

        # Retrieve
        source_docs, graph_context = self.retriever.retrieve(
            question, k=self.retrieval_k
        )

        # Format document context
        doc_context_parts = []
        for i, doc in enumerate(source_docs, 1):
            source = doc.metadata.get("source", "unknown")
            doc_context_parts.append(
                f"[Document {i} | Source: {source}]\n{doc.page_content}"
            )
        document_context = "\n\n".join(doc_context_parts) or "No documents retrieved."

        # Build prompt
        prompt = QA_PROMPT.format(
            graph_context=graph_context,
            document_context=document_context,
            question=question,
        )

        # Generate answer
        answer = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": answer,
            "source_docs": source_docs,
            "graph_context": graph_context,
        }

    # ------------------------------------------------------------------
    # Batch Q&A (for evaluation datasets)
    # ------------------------------------------------------------------
    def batch_ask(self, questions: list[str]) -> list[dict]:
        """Runs ask() for a list of questions. Returns list of result dicts."""
        return [self.ask(q) for q in questions]

    # ------------------------------------------------------------------
    # Pipeline info
    # ------------------------------------------------------------------
    def info(self) -> dict:
        """Returns a summary of all pipeline components."""
        info: dict = {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "retrieval_k": self.retrieval_k,
            "built": self._is_built,
        }
        if self._is_built:
            info["graph_stats"] = self.graph_builder.graph_stats()
            info["vector_store_stats"] = self.vector_store_manager.stats()
        return info

    # ------------------------------------------------------------------
    # Visualization shortcut
    # ------------------------------------------------------------------
    def visualize_graph(self, output_path: str = "data/knowledge_graph.png") -> None:
        """Renders and saves the knowledge graph visualization."""
        self._check_built()
        self.graph_builder.visualize(output_path=output_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _check_built(self) -> None:
        if not self._is_built:
            raise RuntimeError(
                "Pipeline not built. Call build(corpus_path) first."
            )