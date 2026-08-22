import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.loader import fetch_posts
from ingestion.splitter import split_posts

# Load environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    Initializes and returns an embedding model based on provider configuration (.env or arguments).

    Supports:
    - 'ollama' (e.g. nomic-embed-text)
    - 'huggingface' (e.g. sentence-transformers/all-MiniLM-L6-v2)

    Args:
        provider: 'ollama' or 'huggingface' (defaults to EMBEDDING_PROVIDER in .env).
        model_name: Name of the model (defaults to model specified in .env).

    Returns:
        Embeddings: LangChain Embeddings instance.
    """
    selected_provider = (provider or os.getenv("EMBEDDING_PROVIDER", "ollama")).lower().strip()

    if selected_provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError("Please install langchain-ollama: `uv pip install langchain-ollama`")

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        selected_model = model_name or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=selected_model, base_url=base_url)

    elif selected_provider in ["huggingface", "hf", "sentence-transformers"]:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("Please install langchain-huggingface: `uv pip install langchain-huggingface`")

        selected_model = model_name or os.getenv(
            "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceEmbeddings(model_name=selected_model)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{selected_provider}'. "
            "Please use 'ollama' or 'huggingface'."
        )


def get_vector_store(
    embedding_model: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Initializes and returns a persistent ChromaDB vector store instance.

    Args:
        embedding_model: Pre-initialized Embeddings instance (or initialized automatically).
        persist_directory: Local directory for ChromaDB storage (defaults to .env).
        collection_name: Name of the Chroma collection (defaults to .env).

    Returns:
        Chroma: Persistent LangChain Chroma vector store.
    """
    model = embedding_model or get_embedding_model()
    persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    coll_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "wordpress_rag")

    persist_path = Path(persist_dir).resolve()
    persist_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=coll_name,
        embedding_function=model,
        persist_directory=str(persist_path),
    )
    return vector_store


def embed_single_chunk(
    chunk: Union[str, Document],
    embedding_model: Optional[Embeddings] = None,
    vector_store: Optional[Chroma] = None,
    doc_id: Optional[str] = None,
) -> List[float]:
    """
    Raw function that processes ONE chunk at a time:
    1. Computes its vector embedding using the embedding model.
    2. Stores the chunk's text and metadata into the Chroma vector store (if provided).
    3. Returns the embedding vector as a list of floats.

    Args:
        chunk: A single text string or LangChain Document object.
        embedding_model: Optional Embeddings model instance.
        vector_store: Optional Chroma vector store instance to persist the chunk.
        doc_id: Optional unique identifier for the chunk in the vector store.

    Returns:
        List[float]: High-dimensional vector embedding of the chunk.
    """
    model = embedding_model or get_embedding_model()

    # Extract text content and metadata
    if isinstance(chunk, Document):
        text_to_embed = chunk.page_content
        metadata = chunk.metadata or {}
    else:
        text_to_embed = str(chunk)
        metadata = {}

    if not text_to_embed.strip():
        raise ValueError("Cannot embed empty or whitespace-only chunk.")

    # 1. Generate the embedding vector
    vector = model.embed_query(text_to_embed)

    # 2. Store document and metadata in Vector Store if provided
    if vector_store is not None:
        chunk_id = doc_id or str(metadata.get("chunk_id", uuid.uuid4()))
        doc = Document(page_content=text_to_embed, metadata=metadata)
        vector_store.add_documents(documents=[doc], ids=[chunk_id])

    return vector


def embed_chunks(
    chunks: List[Union[str, Document]],
    embedding_model: Optional[Embeddings] = None,
    vector_store: Optional[Chroma] = None,
) -> List[List[float]]:
    """
    Takes all chunks at once, computes their embeddings, and stores them in the vector store.
    Internally iterates in a loop and calls `embed_single_chunk` for each chunk.

    Args:
        chunks: List of text strings or LangChain Document objects.
        embedding_model: Optional pre-initialized Embeddings model.
        vector_store: Optional Chroma vector store instance to store all chunks.

    Returns:
        List[List[float]]: List of embedding vectors corresponding to each input chunk.
    """
    if not chunks:
        print("No chunks provided for embedding.")
        return []

    # Initialize model and vector store once to avoid re-instantiation overhead in loop
    model = embedding_model or get_embedding_model()
    store = vector_store

    all_vectors: List[List[float]] = []

    print(f"Processing and storing {len(chunks)} chunk(s) one-by-one into ChromaDB...")

    for index, chunk in enumerate(chunks, 1):
        try:
            # Generate deterministic or unique chunk ID
            metadata = chunk.metadata if isinstance(chunk, Document) else {}
            post_id = metadata.get("post_id", "post")
            chunk_idx = metadata.get("chunk_id", index)
            unique_id = f"post_{post_id}_chunk_{chunk_idx}_{index}"

            # Process single chunk: generate vector & persist in vector store
            vector = embed_single_chunk(
                chunk=chunk,
                embedding_model=model,
                vector_store=store,
                doc_id=unique_id,
            )
            all_vectors.append(vector)
            print(f"  -> Embedded & stored chunk {index}/{len(chunks)} (ID: {unique_id}, Dim: {len(vector)})")
        except Exception as e:
            print(f"  -> Error processing chunk {index}: {e}")
            raise e

    print(f"Successfully embedded and persisted {len(all_vectors)} chunks in ChromaDB.\n")
    return all_vectors


### Live testing
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Embedding & ChromaDB Storage Pipeline")
    print("=" * 60)

    # 1. Ingest posts
    print("\n[Step 1] Loading posts...")
    posts = fetch_posts(post_type="posts")

    # 2. Split posts
    print("\n[Step 2] Splitting posts into chunks...")
    chunks = split_posts(posts=posts, chunk_size=150, chunk_overlap=15)

    if chunks:
        # Initialize embedding model and persistent Chroma vector store
        model = get_embedding_model()
        vector_store = get_vector_store(embedding_model=model)

        # 3. Test single chunk embedding & storage
        print("\n[Step 3] Testing `embed_single_chunk` with ChromaDB storage on first chunk:")
        sample_chunk = chunks[0]
        single_vector = embed_single_chunk(
            chunk=sample_chunk,
            embedding_model=model,
            vector_store=vector_store,
            doc_id="test_single_chunk_0",
        )
        print(f"Sample Chunk Text Snippet : {sample_chunk.page_content[:80]}...")
        print(f"Single Vector Dimensions  : {len(single_vector)}")
        print(f"First 5 Vector Values     : {single_vector[:5]}")

        # 4. Test all chunks embedding and storage via loop
        print("\n[Step 4] Storing all chunks into ChromaDB via `embed_chunks`:")
        all_vectors = embed_chunks(
            chunks=chunks,
            embedding_model=model,
            vector_store=vector_store,
        )

        # 5. Verify retrieval with a test similarity search
        print("\n[Step 5] Testing Similarity Search from ChromaDB:")
        query = "The UltraBlend 3000 has a 2-year warranty."
        print(f"Query: '{query}'")
        search_results = vector_store.similarity_search(query, k=5)

        for i, res in enumerate(search_results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Title   : {res.metadata.get('title')}")
            print(f"Link    : {res.metadata.get('link')}")
            print(f"Post ID : {res.metadata.get('post_id')}")
            print(f"Content : {res.page_content}")
            print("-" * 40)
