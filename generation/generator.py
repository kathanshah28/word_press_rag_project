import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import retrieve_relevant_chunks, retrieve_with_scores

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


def preprocess_query(query: str) -> str:
    """
    Cleans and normalizes the user query string:
    - Strips leading/trailing whitespaces.
    - Replaces consecutive spaces/tabs/newlines with a single space.
    - Normalizes question punctuation.
    """
    if not query:
        return ""
    normalized = re.sub(r"\s+", " ", query).strip()
    return normalized


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
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Complete RAG pipeline with optional Debug Mode:
    1. Normalizes and preprocesses the user query.
    2. Retrieves top-k matching chunks from ChromaDB with similarity scores.
    3. Formats retrieved context into the system prompt with strict guardrails.
    4. Prints debug details if debug=True.
    5. Invokes the Ollama LLM to generate the grounded response.

    Args:
        query: User input query string.
        top_k: Number of relevant chunks to retrieve (default: 5).
        llm: Optional pre-configured ChatOllama instance.
        debug: If True, prints normalized query, retrieved chunks with scores, and final prompt.

    Returns:
        Dict[str, Any]: Dictionary containing 'query', 'normalized_query', 'answer', 'sources', and 'raw_documents'.
    """
    # 1. Preprocess & normalize query
    normalized_query = preprocess_query(query)
    if not normalized_query:
        return {
            "query": query,
            "normalized_query": "",
            "answer": "Please provide a valid question.",
            "sources": [],
            "raw_documents": [],
        }

    # 2. Retrieve top-k documents with similarity scores from ChromaDB
    scored_docs: List[Tuple[Document, float]] = retrieve_with_scores(query=normalized_query, top_k=top_k)
    retrieved_docs = [doc for doc, _ in scored_docs]

    # 3. Format context block for the LLM
    context_str = format_context_documents(retrieved_docs)

    # 4. Render the final prompt
    chat_llm = llm or get_llm()
    rendered_messages = RAG_PROMPT_TEMPLATE.format_messages(
        context=context_str,
        question=normalized_query,
    )

    # ==========================================================================
    # DEBUG MODE OUTPUT
    # ==========================================================================
    if debug:
        print("\n" + "=" * 75)
        print("🛠️  [DEBUG MODE ENABLED]")
        print("=" * 75)

        # 1. Preprocessed Query
        print("\n🔍 1. PREPROCESSED / NORMALIZED QUERY:")
        print(f"   Original : \"{query}\"")
        print(f"   Normalized: \"{normalized_query}\"")

        # 2. Top-k Retrieved Passages with Scores and Titles
        print(f"\n📚 2. TOP-{len(scored_docs)} RETRIEVED PASSAGES (with Similarity Scores):")
        for rank, (doc, score) in enumerate(scored_docs, 1):
            title = doc.metadata.get("title", "Untitled")
            link = doc.metadata.get("link", "N/A")
            post_id = doc.metadata.get("post_id", "N/A")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            snippet = doc.page_content.replace("\n", " ")[:120]

            print(f"   [{rank}] Score (Distance): {score:.4f} | Post ID: {post_id} | Title: \"{title}\"")
            print(f"       Link    : {link}")
            print(f"       Chunk ID: {chunk_id}")
            print(f"       Content : \"{snippet}...\"")
            print("   " + "-" * 70)

        # 3. Final Prompt sent to LLM
        print("\n📝 3. FINAL PROMPT RENDERED TO LLM:")
        print("   " + "-" * 70)
        for msg in rendered_messages:
            msg_type = msg.type.upper()
            print(f"   [{msg_type} MESSAGE]:\n{msg.content}\n")
        print("   " + "-" * 70)
        print("=" * 75 + "\n")

    # 5. Invoke LLM chain
    rag_chain = RAG_PROMPT_TEMPLATE | chat_llm | StrOutputParser()
    response_text = rag_chain.invoke(
        {
            "context": context_str,
            "question": normalized_query,
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
        "normalized_query": normalized_query,
        "answer": response_text.strip(),
        "sources": sources,
        "raw_documents": retrieved_docs,
    }


# ==============================================================================
# Interactive Command Line Interface (CLI)
# ==============================================================================


def interactive_cli(debug_mode: bool = False):
    """
    Starts an interactive command-line session for asking questions to WordPress RAG.

    Args:
        debug_mode: Whether debug mode is active initially.
    """
    llm = get_llm()
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    debug = debug_mode

    print("=" * 75)
    print("  WordPress Knowledge Base - RAG Q&A Assistant")
    print(f"  Powered by Ollama ({model_name}) + ChromaDB")
    print(f"  Debug Mode: {'[ON]' if debug else '[OFF]'} (Type ':debug' to toggle)")
    print("=" * 75)
    print("Type your question below, or type 'exit' or 'quit' to close.\n")

    while True:
        try:
            user_query = input("Ask a question > ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "q", ":q"]:
                print("\nGoodbye!")
                break

            # Toggle debug mode dynamically in interactive CLI
            if user_query.lower() in [":debug", "/debug", "debug"]:
                debug = not debug
                status_str = "ENABLED [ON]" if debug else "DISABLED [OFF]"
                print(f"\n🔄 Debug Mode is now: {status_str}\n")
                continue

            if not debug:
                print("\nSearching site content and generating response...")

            result = generate_answer(query=user_query, top_k=5, llm=llm, debug=debug)

            print("\n" + "-" * 75)
            print("ANSWER:")
            print(result["answer"])
            print("-" * 75)

            if result["sources"] and "I couldn’t find that in the site content" not in result["answer"]:
                print("Sources:")
                for src in result["sources"]:
                    print(f"  • {src['title']}: {src['link']}")
                print("-" * 75)

            print()

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as err:
            print(f"\n[Error occurred]: {err}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WordPress RAG Generation Assistant")
    parser.add_argument(
        "query",
        nargs="*",
        help="Optional question to ask directly (e.g. python generator.py 'What is the warranty?')",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable Debug Mode to inspect preprocessed query, top-k chunks with scores, and final prompt",
    )
    args = parser.parse_args()

    if args.query:
        single_query = " ".join(args.query)
        res = generate_answer(query=single_query, top_k=5, debug=args.debug)
        print("ANSWER:")
        print(res["answer"])
        if res["sources"] and "I couldn’t find that in the site content" not in res["answer"]:
            print("\nSources:")
            for s in res["sources"]:
                print(f"- {s['title']} ({s['link']})")
    else:
        interactive_cli(debug_mode=args.debug)
