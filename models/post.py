"""
Pydantic schemas and data models for WordPress RAG pipeline.
"""

from typing import Any, Dict, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


class CleanedPost(BaseModel):
    """
    Pydantic data model representing a cleaned and validated WordPress post
    ready for chunking, text embedding, and vector storage.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    post_id: int = Field(..., description="Unique WordPress post ID")
    title: str = Field(..., description="Cleaned, plain-text post title")
    link: str = Field(..., description="Permalink URL of the post")
    content: str = Field(..., description="Cleaned, plain-text main body content")

    def to_document(self) -> Document:
        """
        Converts the CleanedPost instance into a LangChain Document object
        with formatted page_content and structured metadata.

        Returns:
            Document: LangChain Document ready for embeddings & vector store.
        """
        page_content = f"# {self.title}\n\n{self.content}" if self.title else self.content
        metadata: Dict[str, Any] = {
            "post_id": self.post_id,
            "title": self.title,
            "link": self.link,
            "source": self.link,
        }
        return Document(page_content=page_content, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Returns standard dictionary representation."""
        return {
            "post_id": self.post_id,
            "title": self.title,
            "link": self.link,
            "content": self.content,
        }
