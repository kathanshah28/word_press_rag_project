import sys
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.post import CleanedPost
from ingestion.loader import fetch_posts


def split_posts(
    posts: Union[List[CleanedPost], List[Document]],
    chunk_size: int = 150,
    chunk_overlap: int = 15,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Splits cleaned WordPress posts into smaller chunks using LangChain's
    RecursiveCharacterTextSplitter while strictly preserving all document metadata.

    Args:
        posts: List of CleanedPost Pydantic objects or LangChain Document objects.
        chunk_size: Maximum character length for each chunk (default: 150).
        chunk_overlap: Number of overlapping characters between chunks (default: 15).
        separators: Optional custom separators list (default: ["\\n\\n", "\\n", ". ", " ", ""]).

    Returns:
        List[Document]: List of chunked LangChain Document objects with preserved metadata.
    """
    if not posts:
        print("No posts provided for splitting.")
        return []

    # 1. Convert Pydantic CleanedPost objects to LangChain Documents if needed
    documents: List[Document] = []
    for item in posts:
        if isinstance(item, CleanedPost):
            documents.append(item.to_document())
        elif isinstance(item, Document):
            documents.append(item)
        elif isinstance(item, dict):
            # If passed as raw dict, create Document
            page_content = f"# {item.get('title', '')}\n\n{item.get('content', '')}".strip()
            metadata = {
                "post_id": item.get("post_id") or item.get("id"),
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "source": item.get("link", ""),
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
        else:
            raise TypeError(f"Unsupported item type for splitting: {type(item)}")

    # 2. Configure RecursiveCharacterTextSplitter
    custom_separators = separators or ["\n\n", "\n", ". ", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=custom_separators,
        length_function=len,
        is_separator_regex=False,
    )

    # 3. Perform document splitting (preserves metadata automatically)
    chunks = text_splitter.split_documents(documents)

    # 4. Add chunk index metadata for tracking
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(
        f"Splitting completed: {len(documents)} original document(s) -> {len(chunks)} chunk(s) "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks


### Live testing
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Live Ingestion & Splitting Test")
    print("=" * 60)

    # 1. Fetch posts using ingestion/loader.py
    print("\n[Step 1] Loading posts from WordPress...")
    posts = fetch_posts(post_type="posts")

    # 2. Split posts using RecursiveCharacterTextSplitter
    print("\n[Step 2] Splitting posts into chunks...")
    chunks = split_posts(posts=posts, chunk_size=150, chunk_overlap=15)

    # 3. Display split results and verify preserved metadata
    print("\n[Step 3] Verification & Chunk Inspection:")
    print(f"Total Chunks Generated: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- [Chunk {i} / {len(chunks)}] ---")
        print(f"Post ID       : {chunk.metadata.get('post_id')}")
        print(f"Title         : {chunk.metadata.get('title')}")
        print(f"Link          : {chunk.metadata.get('link')}")
        print(f"Chunk ID      : {chunk.metadata.get('chunk_id')}")
        print(f"Character Len : {len(chunk.page_content)}")
        print(f"Preserved Meta: {chunk.metadata}")
        print(f"Chunk Content :\n{chunk.page_content}")
        print("-" * 60)
