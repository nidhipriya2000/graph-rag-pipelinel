# Graph RAG Pipeline

> **Retrieval-Augmented Generation with Knowledge Graphs, NetworkX, LangChain, and RAGAS — fully local via Ollama.**

A production-style pipeline that extracts a knowledge graph from unstructured text, performs hybrid (graph + vector) retrieval, and evaluates answer quality using the RAGAS benchmarking framework — no cloud API keys required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TEXT CORPUS                              │
│                  (unstructured .txt files)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
          ▼                                    ▼
┌──────────────────┐                ┌──────────────────────┐
│  KnowledgeGraph  │                │   VectorStore         │
│  Builder         │                │   Manager             │
│                  │                │                       │
│  Ollama LLM      │                │  Ollama Embeddings    │
│  → Extract       │                │  (nomic-embed-text)   │
│    (subj,rel,obj)│                │  → Chunk + Embed      │
│  → NetworkX      │                │  → Chroma DB          │
│    DiGraph       │                └──────────┬────────────┘
└────────┬─────────┘                           │
         │                                     │
         └─────────────────┬───────────────────┘
                           ▼
               ┌───────────────────────┐
               │   HybridRetriever     │
               │                       │
               │  Vector Search        │
               │  + Graph Expansion    │
               │  + RRF Re-ranking     │
               └──────────┬────────────┘
                          │
                          ▼
               ┌───────────────────────┐
               │   GraphRAGPipeline    │
               │                       │
               │  Graph Context +      │
               │  Doc Context +        │
               │  LangChain QA Chain   │
               └──────────┬────────────┘
                          │
                          ▼
               ┌───────────────────────┐
               │   RAGASEvaluator      │
               │                       │
               │  faithfulness         │
               │  answer_relevancy     │
               │  context_recall       │
               │  context_precision    │
               └───────────────────────┘
```

---

## Key Features

| Feature | Details |
|---|---|
| **Knowledge Graph Construction** | LLM-powered (subject, relation, object) triple extraction → NetworkX DiGraph |
| **Hybrid Retrieval** | Vector similarity search fused with graph neighbourhood expansion via Reciprocal Rank Fusion (RRF) |
| **Local-First** | Entirely powered by Ollama — no OpenAI, Anthropic, or cloud API keys needed |
| **RAGAS Evaluation** | Automated benchmarking across 4 retrieval quality metrics |
| **Graph Visualisation** | matplotlib rendering of the extracted knowledge graph |
| **Persistence** | Chroma vector store + JSON graph cache for fast reloads |
| **Clean CLI** | Interactive Q&A session and single-question mode |

---

## Quickstart

### 1. Prerequisites

Install [Ollama](https://ollama.com) and pull the required models:

```bash
ollama pull llama3.2          # LLM for generation + triple extraction
ollama pull nomic-embed-text  # Embedding model for vector store
```

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
# Interactive Q&A
python main.py

# Ask a single question
python main.py --question "Who developed BERT and on which datasets was it trained?"

# Force rebuild (re-extract graph, re-embed all documents)
python main.py --rebuild

# Show pipeline stats
python main.py --info

# Visualize the knowledge graph
python main.py --visualize
```

### 4. Run RAGAS evaluation

```bash
python evaluate.py
```

**Example output:**
```
==========================================================
  GRAPH RAG PIPELINE — RAGAS EVALUATION REPORT
==========================================================
  Metric                    Score  Bar
----------------------------------------------------------
  faithfulness              0.847  [████████████████░░░░] ✓
  answer_relevancy          0.821  [████████████████░░░░] ✓
  context_recall            0.793  [███████████████░░░░░] ✓
  context_precision         0.764  [███████████████░░░░░] ✓
  composite                 0.806  [████████████████░░░░] ✓
==========================================================
  Evaluated on 8 questions
==========================================================
```

---

## Project Structure

```
graph_rag_pipeline/
├── data/
│   ├── corpus.txt               # Sample 10-document AI/ML corpus
│   ├── knowledge_graph.json     # Cached graph (auto-generated)
│   ├── knowledge_graph.png      # Graph visualisation (auto-generated)
│   ├── chroma_db/               # Persisted vector store (auto-generated)
│   └── evaluation_report.json  # RAGAS results (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── graph_builder.py         # Triple extraction + NetworkX graph
│   ├── vector_store.py          # Chroma + Ollama embeddings
│   ├── retriever.py             # Hybrid retrieval + RRF fusion
│   ├── pipeline.py              # LangChain Q&A orchestration
│   └── evaluator.py             # RAGAS evaluation suite
│
├── main.py                      # CLI entry point
├── evaluate.py                  # Standalone evaluation runner
├── requirements.txt
└── README.md
```

---

## Component Deep-Dives

### Knowledge Graph Builder (`src/graph_builder.py`)

Extracts structured knowledge from raw text using a locally-running LLM:

```python
from src.graph_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(model="llama3.2")
documents = builder.load_corpus("data/corpus.txt")
graph = builder.build_graph(documents)

# Query the graph
print(builder.get_entity_context("BERT"))
# Knowledge graph context for 'BERT':
#   BERT —[developed by]→ Google AI
#   BERT —[trained on]→ Wikipedia corpus
#   BERT —[uses]→ WordPiece tokenization

print(builder.graph_stats())
# {'nodes': 87, 'edges': 124, 'density': 0.0165, ...}
```

The LLM is prompted to return JSON arrays of `(subject, relation, object)` triples. The graph is persisted as node-link JSON for fast reloads.

---

### Hybrid Retriever (`src/retriever.py`)

Combines two retrieval signals:

1. **Vector search** — semantic similarity via Chroma + nomic-embed-text embeddings
2. **Graph expansion** — identifies entities in the query, traverses their graph neighbourhood, and fetches vector-store chunks for related entities

Results are merged using **Reciprocal Rank Fusion (RRF)**, a parameter-free score fusion method that rewards documents appearing near the top of multiple ranked lists.

```python
from src.retriever import HybridRetriever

retriever = HybridRetriever(graph_builder, vector_store_manager)
docs, graph_context = retriever.retrieve("How does BERT use attention?", k=5)
```

---

### RAGAS Evaluation (`src/evaluator.py`)

RAGAS metrics explained:

| Metric | What it measures | How it's computed |
|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? | LLM checks each claim in the answer against context |
| **Answer Relevancy** | Does the answer address the question? | Embedding similarity between question and answer |
| **Context Recall** | Is the ground truth covered by retrieved context? | LLM checks each ground-truth sentence against context |
| **Context Precision** | Is retrieved context mostly signal, not noise? | Fraction of context chunks that are actually relevant |

All four metrics use the **local Ollama LLM as judge** — no external API calls.

---

## Configuration

Key parameters can be adjusted in `main.py` and `evaluate.py`:

| Parameter | Default | Effect |
|---|---|---|
| `llm_model` | `llama3.2` | Ollama model used for generation and triple extraction |
| `embedding_model` | `nomic-embed-text` | Ollama model used for embeddings |
| `retrieval_k` | `5` | Number of chunks retrieved per query |
| `chunk_size` | `512` | Token size for document splitting |
| `chunk_overlap` | `64` | Overlap between consecutive chunks |
| `graph_hop_depth` | `1` | Neighbourhood depth for graph expansion |

---

## Iterative Improvement Workflow

The RAGAS evaluator includes `suggest_improvements()` that maps low metric scores to concrete fixes:

| Low metric | Suggested fix |
|---|---|
| Faithfulness < 0.7 | Lower LLM temperature; strengthen context-grounding in prompt |
| Answer relevancy < 0.7 | Add query rewriting; increase chunk overlap |
| Context recall < 0.7 | Increase `retrieval_k`; reduce chunk size; deeper graph hops |
| Context precision < 0.7 | Reduce `retrieval_k`; add cross-encoder re-ranker |

---

## Adding Your Own Documents

The corpus uses a simple `===DOCUMENT: <name>===` delimiter format:

```
===DOCUMENT: my_paper===
Your document text goes here. The pipeline will automatically
extract entities, build graph edges, and embed chunks for retrieval.

===DOCUMENT: another_paper===
More text...
```

Point the pipeline at your file:

```python
pipeline.build("path/to/your/corpus.txt", force_rebuild_graph=True)
```

---

## Tech Stack

- **[Ollama](https://ollama.com)** — local LLM inference (llama3.2, nomic-embed-text)
- **[NetworkX](https://networkx.org)** — knowledge graph construction and traversal
- **[LangChain](https://langchain.com)** — retrieval chain orchestration
- **[Chroma](https://trychroma.com)** — local vector database
- **[RAGAS](https://docs.ragas.io)** — RAG evaluation framework
- **[Matplotlib](https://matplotlib.org)** — graph visualization

---

## License

MIT
