"""
graph_builder.py
----------------
Extracts entities and relationships from text documents using a local Ollama LLM,
then constructs a NetworkX knowledge graph from the extracted triples.
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template for triple extraction
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a knowledge graph extraction engine.
Given the text below, extract a list of factual (subject, relation, object) triples.

Rules:
- Subject and object must be named entities (people, organizations, models, concepts, tools).
- Relation must be a short verb phrase (e.g., "developed by", "uses", "is part of", "introduced").
- Extract 5–12 triples per document.
- Return ONLY a JSON array, no explanation, no markdown fences.

Format:
[
  {{"subject": "...", "relation": "...", "object": "..."}},
  ...
]

Text:
{text}
"""


# ---------------------------------------------------------------------------
# KnowledgeGraphBuilder
# ---------------------------------------------------------------------------
class KnowledgeGraphBuilder:
    """
    Builds a NetworkX DiGraph from a text corpus by extracting (subject, relation, object)
    triples via a local Ollama LLM.
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.llm = OllamaLLM(model=model, base_url=base_url, temperature=0)
        self.graph = nx.DiGraph()
        self._documents: list[Document] = []

    # ------------------------------------------------------------------
    # Corpus loading
    # ------------------------------------------------------------------
    def load_corpus(self, path: str) -> list[Document]:
        """
        Parses a corpus file segmented by ===DOCUMENT: <name>=== headers.
        Returns a list of LangChain Document objects.
        """
        corpus_path = Path(path)
        raw = corpus_path.read_text(encoding="utf-8")

        sections = re.split(r"===DOCUMENT:\s*(.+?)===", raw)
        documents: list[Document] = []

        # sections[0] is text before first header (empty), then alternating name/content
        for i in range(1, len(sections), 2):
            doc_name = sections[i].strip()
            doc_text = sections[i + 1].strip() if i + 1 < len(sections) else ""
            if doc_text:
                documents.append(Document(page_content=doc_text, metadata={"source": doc_name}))

        self._documents = documents
        logger.info("Loaded %d documents from corpus.", len(documents))
        return documents

    # ------------------------------------------------------------------
    # Triple extraction
    # ------------------------------------------------------------------
    def extract_triples(self, document: Document) -> list[dict]:
        """
        Calls the LLM to extract (subject, relation, object) triples from one document.
        Returns a list of triple dicts, or empty list on parse failure.
        """
        prompt = EXTRACTION_PROMPT.format(text=document.page_content)
        response = self.llm.invoke(prompt)

        # Strip markdown fences if model includes them
        cleaned = re.sub(r"```(?:json)?|```", "", response).strip()

        try:
            triples = json.loads(cleaned)
            if not isinstance(triples, list):
                raise ValueError("Expected a JSON array.")
            return triples
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse triples from '%s': %s", document.metadata["source"], exc
            )
            return []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_graph(self, documents: Optional[list[Document]] = None) -> nx.DiGraph:
        """
        Iterates over documents, extracts triples, and adds them to the NetworkX graph.
        Nodes store the entity name; edges store the relation and source document.
        """
        docs = documents or self._documents
        if not docs:
            raise ValueError("No documents loaded. Call load_corpus() first.")

        total_triples = 0

        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            logger.info("Extracting triples from: %s", source)

            triples = self.extract_triples(doc)
            logger.info("  → %d triples extracted.", len(triples))

            for triple in triples:
                subj = triple.get("subject", "").strip()
                rel  = triple.get("relation", "").strip()
                obj  = triple.get("object", "").strip()

                if not (subj and rel and obj):
                    continue

                # Add nodes with type metadata
                self.graph.add_node(subj, node_type="entity")
                self.graph.add_node(obj,  node_type="entity")

                # Add (or update) directed edge
                if self.graph.has_edge(subj, obj):
                    # Accumulate relations if the edge already exists
                    existing = self.graph[subj][obj]["relations"]
                    if rel not in existing:
                        existing.append(rel)
                else:
                    self.graph.add_edge(
                        subj, obj,
                        relations=[rel],
                        sources=[source]
                    )

                total_triples += 1

        logger.info(
            "Graph built: %d nodes, %d edges from %d triples.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            total_triples,
        )
        return self.graph

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def get_neighbors(self, entity: str, depth: int = 1) -> list[str]:
        """
        Returns all nodes reachable from `entity` within `depth` hops (both directions).
        """
        entity_lower = entity.lower()
        matched = [n for n in self.graph.nodes if n.lower() == entity_lower]
        if not matched:
            return []

        node = matched[0]
        subgraph_nodes = nx.ego_graph(self.graph.to_undirected(), node, radius=depth).nodes()
        return [n for n in subgraph_nodes if n != node]

    def get_entity_context(self, entity: str) -> str:
        """
        Builds a textual summary of an entity's direct connections in the graph.
        Useful for injecting graph context into the LLM prompt.
        """
        entity_lower = entity.lower()
        matched = [n for n in self.graph.nodes if n.lower() == entity_lower]
        if not matched:
            return f"No graph context found for '{entity}'."

        node = matched[0]
        lines = [f"Knowledge graph context for '{node}':"]

        # Outgoing edges
        for _, obj, data in self.graph.out_edges(node, data=True):
            rels = ", ".join(data.get("relations", []))
            lines.append(f"  {node} —[{rels}]→ {obj}")

        # Incoming edges
        for subj, _, data in self.graph.in_edges(node, data=True):
            rels = ", ".join(data.get("relations", []))
            lines.append(f"  {subj} —[{rels}]→ {node}")

        return "\n".join(lines)

    def graph_stats(self) -> dict:
        """Returns basic statistics about the constructed graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            "top_nodes_by_degree": sorted(
                self.graph.degree(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_graph(self, path: str = "data/knowledge_graph.json") -> None:
        """Saves the graph as node-link JSON, compatible with all NetworkX versions."""
        try:
            data = nx.node_link_data(self.graph, edges="edges")
        except TypeError:
            data = nx.node_link_data(self.graph)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Graph saved to %s", path)

    def load_graph(self, path: str = "data/knowledge_graph.json") -> nx.DiGraph:
        """Loads a graph from node-link JSON, compatible with all NetworkX versions."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            if "links" in data and "edges" not in data:
                self.graph = nx.node_link_graph(data, edges="links")
            else:
                self.graph = nx.node_link_graph(data, edges="edges")
        except TypeError:
            self.graph = nx.node_link_graph(data)
        logger.info(
            "Graph loaded: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def visualize(
        self,
        max_nodes: int = 40,
        output_path: str = "data/knowledge_graph.png",
        figsize: tuple = (18, 12),
    ) -> None:
        """
        Renders the knowledge graph using matplotlib.
        Limits to the top `max_nodes` nodes by degree for readability.
        """
        graph = self.graph

        # Subset to top-N nodes by degree
        if graph.number_of_nodes() > max_nodes:
            top_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_names = [n for n, _ in top_nodes]
            graph = graph.subgraph(top_node_names)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#0f0f1a")
        fig.patch.set_facecolor("#0f0f1a")

        pos = nx.spring_layout(graph, k=2.5, seed=42)

        # Node degree-based sizing
        degrees = dict(graph.degree())
        node_sizes = [300 + degrees.get(n, 1) * 120 for n in graph.nodes()]
        node_colors = ["#7c3aed" if degrees.get(n, 0) > 3 else "#2563eb" for n in graph.nodes()]

        nx.draw_networkx_nodes(
            graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph, pos,
            font_size=7,
            font_color="white",
            font_weight="bold",
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph, pos,
            edge_color="#4b5563",
            arrows=True,
            arrowsize=12,
            width=0.8,
            alpha=0.7,
            ax=ax,
        )

        # Edge labels (first relation only to avoid clutter)
        edge_labels = {
            (u, v): data["relations"][0]
            for u, v, data in graph.edges(data=True)
            if data.get("relations")
        }
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=6,
            font_color="#9ca3af",
            ax=ax,
        )

        hub = mpatches.Patch(color="#7c3aed", label="Hub node (degree > 3)")
        reg = mpatches.Patch(color="#2563eb", label="Regular node")
        ax.legend(handles=[hub, reg], facecolor="#1f2937", labelcolor="white", fontsize=9)
        ax.set_title("Knowledge Graph", color="white", fontsize=14, pad=15)
        ax.axis("off")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info("Graph visualization saved to %s", output_path)