import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding.embedder import get_embedding_model, get_vector_store

# Load environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    vector_store: Optional[Chroma] = None,
) -> List[Document]:
    """
    Retrieves the top-k most relevant chunks from the persistent ChromaDB vector store
    based on semantic similarity to the given user query.

    Args:
        query: The user question or search query string.
        top_k: Number of best-matching results to return (default: 5).
        filter_metadata: Optional metadata filter dict for ChromaDB (e.g. {"post_id": 5}).
        vector_store: Optional pre-initialized Chroma instance (initialized if None).

    Returns:
        List[Document]: List of top-k LangChain Document objects matching the query.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")

    store = vector_store or get_vector_store()

    # Query ChromaDB for top_k similarity matches
    results = store.similarity_search(
        query=query,
        k=top_k,
        filter=filter_metadata,
    )

    return results


def retrieve_with_scores(
    query: str,
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    vector_store: Optional[Chroma] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieves the top-k most relevant chunks from ChromaDB along with their
    similarity distance scores (lower distance indicates higher similarity in L2/cosine).

    Args:
        query: The user question or search query string.
        top_k: Number of best-matching results to return (default: 5).
        filter_metadata: Optional metadata filter dict for ChromaDB.
        vector_store: Optional pre-initialized Chroma instance.

    Returns:
        List[Tuple[Document, float]]: List of (Document, score) tuples.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")

    store = vector_store or get_vector_store()

    results_with_scores = store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=filter_metadata,
    )

    return results_with_scores


### Live testing
if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the warranty policy and maintenance procedure for the UltraBlend 3000?"
    )

    print("=" * 70)
    print("WordPress RAG - ChromaDB Top-5 Retrieval Test")
    print("=" * 70)
    print(f"Query: \"{test_query}\"\n")

    # 1. Standard Top-5 Retrieval
    top_docs = retrieve_relevant_chunks(query=test_query, top_k=5)
    print(f"Retrieved {len(top_docs)} document chunk(s):\n")

    for rank, doc in enumerate(top_docs, 1):
        print(f"--- [Rank {rank} / {len(top_docs)}] ---")
        print(f"Title    : {doc.metadata.get('title', 'N/A')}")
        print(f"Post ID  : {doc.metadata.get('post_id', 'N/A')}")
        print(f"Link     : {doc.metadata.get('link', 'N/A')}")
        print(f"Chunk ID : {doc.metadata.get('chunk_id', 'N/A')}")
        print(f"Content  :\n{doc.page_content}")
        print("-" * 70)

    # 2. Retrieval with Similarity Scores
    print("\nScore Breakdown (Lower distance score = higher relevance):")
    scored_results = retrieve_with_scores(query=test_query, top_k=5)
    for rank, (doc, score) in enumerate(scored_results, 1):
        print(f"  #{rank} [Score: {score:.4f}] Post ID: {doc.metadata.get('post_id')} | Title: '{doc.metadata.get('title')}'")
