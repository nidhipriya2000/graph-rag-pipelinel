"""
retriever.py
------------
Hybrid retriever that fuses:
  1. Vector similarity search (semantic retrieval via Chroma)
  2. Graph-based context expansion (NetworkX neighbourhood traversal)

The two result sets are merged, deduplicated, and re-ranked by a simple
reciprocal-rank fusion (RRF) score before being returned to the LLM.
"""

import logging
import re
from typing import Optional

import networkx as nx
from langchain_core.documents import Document

from src.graph_builder import KnowledgeGraphBuilder
from src.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    """
    Merges multiple ranked lists of Documents using Reciprocal Rank Fusion.
    Higher RRF score = more relevant.

    Args:
        ranked_lists: Each inner list is ordered best-first.
        k: Constant that controls the impact of lower-ranked documents.
    Returns:
        Deduplicated list of Documents sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:200]  # Fingerprint by content prefix
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ---------------------------------------------------------------------------
# Entity extraction helper (simple regex + known entity list)
# ---------------------------------------------------------------------------
def _extract_candidate_entities(query: str, graph: nx.DiGraph) -> list[str]:
    """
    Finds graph nodes that are mentioned (case-insensitive) in the query string.
    """
    query_lower = query.lower()
    return [node for node in graph.nodes if node.lower() in query_lower]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Combines vector search with knowledge-graph neighbourhood expansion.

    Usage:
        retriever = HybridRetriever(graph_builder, vector_store_manager)
        docs, graph_ctx = retriever.retrieve("How does BERT use attention?", k=5)
    """

    def __init__(
        self,
        graph_builder: KnowledgeGraphBuilder,
        vector_store: VectorStoreManager,
        graph_hop_depth: int = 1,
    ):
        self.graph_builder = graph_builder
        self.vector_store = vector_store
        self.graph_hop_depth = graph_hop_depth

    # ------------------------------------------------------------------
    # Main retrieve method
    # ------------------------------------------------------------------
    def retrieve(
        self, query: str, k: int = 5
    ) -> tuple[list[Document], str]:
        """
        Retrieves relevant context for `query`.

        Returns:
            (documents, graph_context_string)
            - documents: merged & re-ranked list of Document chunks
            - graph_context_string: textual graph neighbourhood info to
              inject as additional context into the LLM prompt
        """
        # --- 1. Vector search ---
        vector_docs = self._vector_search(query, k=k)

        # --- 2. Graph-guided expansion ---
        graph_docs, graph_context = self._graph_expansion(query, k=k)

        # --- 3. Fuse results ---
        fused = _reciprocal_rank_fusion([vector_docs, graph_docs])
        final_docs = fused[:k]

        logger.debug(
            "Retrieved %d vector docs, %d graph-expanded docs → %d fused.",
            len(vector_docs), len(graph_docs), len(final_docs),
        )
        return final_docs, graph_context

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------
    def _vector_search(self, query: str, k: int) -> list[Document]:
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Graph-based expansion
    # ------------------------------------------------------------------
    def _graph_expansion(
        self, query: str, k: int
    ) -> tuple[list[Document], str]:
        """
        Identifies entities in the query, expands their graph neighbourhood,
        and fetches vector-store chunks for those neighbours.
        Also builds a textual graph context string for the LLM.
        """
        graph = self.graph_builder.graph
        entities = _extract_candidate_entities(query, graph)

        if not entities:
            logger.debug("No graph entities found in query.")
            return [], "No direct graph context found for this query."

        graph_ctx_parts: list[str] = []
        neighbour_queries: list[str] = []

        for entity in entities[:3]:   # limit to top-3 matched entities
            ctx = self.graph_builder.get_entity_context(entity)
            graph_ctx_parts.append(ctx)

            neighbours = self.graph_builder.get_neighbors(
                entity, depth=self.graph_hop_depth
            )
            neighbour_queries.extend(neighbours[:5])  # cap expansion width

        graph_context = "\n\n".join(graph_ctx_parts)

        # Fetch vector chunks for each discovered neighbour
        expanded_docs: list[Document] = []
        seen: set[str] = set()

        for neighbour in neighbour_queries:
            neighbour_docs = self._vector_search(neighbour, k=2)
            for doc in neighbour_docs:
                fp = doc.page_content[:200]
                if fp not in seen:
                    seen.add(fp)
                    expanded_docs.append(doc)

        return expanded_docs, graph_context

    # ------------------------------------------------------------------
    # LangChain-compatible retriever shim
    # ------------------------------------------------------------------
    def as_langchain_retriever(self, k: int = 5):
        """
        Returns an object implementing the LangChain BaseRetriever interface
        so HybridRetriever can slot into any LangChain chain.
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

        hybrid = self

        class _LangChainShim(BaseRetriever):
            def _get_relevant_documents(
                self_inner,
                query: str,
                *,
                run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> list[Document]:
                docs, _ = hybrid.retrieve(query, k=k)
                return docs

        return _LangChainShim()