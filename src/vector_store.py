"""
vector_store.py
---------------
Builds and manages a Chroma vector store from text documents using
Ollama embeddings (nomic-embed-text by default).
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages a Chroma vector store backed by Ollama local embeddings.

    Workflow:
        manager = VectorStoreManager()
        manager.build(documents)          # index documents
        results = manager.similarity_search("BERT model", k=4)
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        persist_directory: str = "data/chroma_db",
        collection_name: str = "graph_rag",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._vectorstore: Optional[Chroma] = None

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------
    def build(self, documents: list[Document], force_rebuild: bool = False) -> Chroma:
        """
        Chunks documents and indexes them into Chroma.
        If the persist directory already exists, loads the existing store
        unless `force_rebuild=True`.
        """
        db_path = Path(self.persist_directory)

        if db_path.exists() and not force_rebuild:
            logger.info("Loading existing Chroma store from %s", self.persist_directory)
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info(
                "Loaded store with %d documents.",
                self._vectorstore._collection.count(),
            )
            return self._vectorstore

        logger.info("Building new Chroma vector store…")
        chunks = self._chunk_documents(documents)
        logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        logger.info(
            "Vector store built and persisted to %s (%d chunks indexed).",
            self.persist_directory,
            len(chunks),
        )
        return self._vectorstore

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Returns the top-k most semantically similar chunks."""
        self._check_store()
        return self._vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Returns (Document, cosine_distance) pairs sorted by relevance."""
        self._check_store()
        return self._vectorstore.similarity_search_with_score(query, k=k)

    def as_retriever(self, k: int = 4):
        """Returns a LangChain-compatible retriever interface."""
        self._check_store()
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Splits documents into overlapping chunks, preserving metadata."""
        chunks: list[Document] = []
        for doc in documents:
            splits = self.splitter.split_text(doc.page_content)
            for i, text in enumerate(splits):
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        return chunks

    def _check_store(self) -> None:
        if self._vectorstore is None:
            raise RuntimeError(
                "Vector store not initialized. Call build() first."
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        self._check_store()
        return {
            "collection": self.collection_name,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_chunks": self._vectorstore._collection.count(),
            "persist_directory": self.persist_directory,
        }