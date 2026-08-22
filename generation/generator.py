import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import retrieve_relevant_chunks

# Load environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ==============================================================================
# Prompt Template & System Instructions
# ==============================================================================

SYSTEM_PROMPT = """\
You are an authoritative and precise WordPress Assistant.
Your task is to answer user questions STRICTLY using the provided site context.

### GUIDELINES (WHAT TO DO):
1. Use ONLY the facts directly mentioned in the "Context Documents" section below.
2. Structure your answer clearly and concisely.
3. If relevant, mention the title and link of the source post.

### STRICT CONSTRAINTS (WHAT NOT TO DO):
1. Do NOT extrapolate, speculate, or bring in outside knowledge that is not in the context.
2. If the provided context does NOT contain enough relevant information to answer the query truthfully and completely, you MUST reply with EXACTLY:
"I couldn’t find that in the site content"
3. NEVER make up or generate false, inaccurate, or unverified information.

---
Context Documents:
{context}
---
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def get_llm(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatOllama:
    """
    Initializes and returns the ChatOllama LLM instance configured from .env.

    Args:
        model_name: Ollama model name (defaults to OLLAMA_MODEL in .env, e.g. llama3.2).
        base_url: Ollama API endpoint (defaults to OLLAMA_BASE_URL in .env).
        temperature: Sampling temperature (0.0 for deterministic factual responses).

    Returns:
        ChatOllama: Initialized ChatOllama client.
    """
    model = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model, base_url=url, temperature=temperature)


def format_context_documents(documents: List[Document]) -> str:
    """
    Formats retrieved LangChain Document objects into a clean, labeled context block.

    Args:
        documents: List of retrieved Document chunks.

    Returns:
        str: Formatted context text with source metadata.
    """
    if not documents:
        return "No relevant documents found."

    formatted_blocks = []
    for idx, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Untitled")
        link = doc.metadata.get("link", "N/A")
        post_id = doc.metadata.get("post_id", "N/A")

        block = (
            f"[Source {idx}]\n"
            f"Title: {title}\n"
            f"Post ID: {post_id}\n"
            f"URL: {link}\n"
            f"Content:\n{doc.page_content.strip()}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def generate_answer(
    query: str,
    top_k: int = 5,
    llm: Optional[ChatOllama] = None,
) -> Dict[str, Any]:
    """
    Complete RAG pipeline:
    1. Retrieves top-k matching chunks from ChromaDB for the query.
    2. Formats retrieved context into the system prompt with strict guardrails.
    3. Invokes the Ollama LLM to generate the grounded response.

    Args:
        query: User input query string.
        top_k: Number of relevant chunks to retrieve (default: 5).
        llm: Optional pre-configured ChatOllama instance.

    Returns:
        Dict[str, Any]: Dictionary containing 'query', 'answer', and 'sources'.
    """
    if not query or not query.strip():
        return {
            "query": query,
            "answer": "Please provide a valid question.",
            "sources": [],
        }

    # 1. Retrieve relevant context from ChromaDB
    retrieved_docs = retrieve_relevant_chunks(query=query, top_k=top_k)

    # 2. Format context
    context_str = format_context_documents(retrieved_docs)

    # 3. Initialize LLM & Chain
    chat_llm = llm or get_llm()
    rag_chain = RAG_PROMPT_TEMPLATE | chat_llm | StrOutputParser()

    # 4. Generate LLM response
    response_text = rag_chain.invoke(
        {
            "context": context_str,
            "question": query,
        }
    )

    # Extract distinct source references
    sources = []
    seen_links = set()
    for doc in retrieved_docs:
        link = doc.metadata.get("link")
        title = doc.metadata.get("title")
        if link and link not in seen_links:
            seen_links.add(link)
            sources.append({"title": title, "link": link})

    return {
        "query": query,
        "answer": response_text.strip(),
        "sources": sources,
        "raw_documents": retrieved_docs,
    }


# ==============================================================================
# Interactive Command Line Interface (CLI)
# ==============================================================================


def interactive_cli():
    """Starts an interactive command-line session for asking questions to WordPress RAG."""
    llm = get_llm()
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")

    print("=" * 70)
    print("  WordPress Knowledge Base - RAG Q&A Assistant")
    print(f"  Powered by Ollama ({model_name}) + ChromaDB")
    print("=" * 70)
    print("Type your question below, or type 'exit' or 'quit' to close.\n")

    while True:
        try:
            user_query = input("Ask a question > ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "q", ":q"]:
                print("\nGoodbye!")
                break

            print("\nSearching site content and generating response...")
            result = generate_answer(query=user_query, top_k=5, llm=llm)

            print("\n" + "-" * 70)
            print("ANSWER:")
            print(result["answer"])
            print("-" * 70)

            if result["sources"] and "I couldn’t find that in the site content" not in result["answer"]:
                print("Sources:")
                for src in result["sources"]:
                    print(f"  • {src['title']}: {src['link']}")
                print("-" * 70)

            print()

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as err:
            print(f"\n[Error occurred]: {err}\n")


if __name__ == "__main__":
    # If a query was passed directly via command line args (e.g. python generator.py "query")
    if len(sys.argv) > 1:
        cli_query = " ".join(sys.argv[1:])
        print(f"User Query: {cli_query}\n")
        res = generate_answer(cli_query)
        print("ANSWER:")
        print(res["answer"])
        print("\nSources:")
        for s in res["sources"]:
            print(f"- {s['title']} ({s['link']})")
    else:
        # Otherwise launch interactive CMD interface
        interactive_cli()
