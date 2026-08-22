# 🚀 WordPress Enterprise RAG System

A production-grade, local **Retrieval-Augmented Generation (RAG)** pipeline for WordPress websites. This project enables natural-language question answering over WordPress posts and pages using **LangChain**, local **Ollama** LLMs (`llama3.2`), local vector embeddings (`nomic-embed-text`), and a persistent **ChromaDB** vector database.

---

## 📌 Architecture & Features

- **Automated WordPress Ingestion**: Connects to the WordPress REST API (`/wp-json/wp/v2/posts`, `pages`), automatically handles pagination, and extracts clean content.
- **Content Sanitization**: Strips WordPress Gutenberg block comments (`<!-- wp:... -->`), shortcodes, decodes HTML entities, and normalizes text.
- **Pydantic Validation**: Validates cleaned data into strongly-typed `CleanedPost` data models.
- **Metadata-Preserving Splitting**: Splits documents into semantically coherent chunks using LangChain's `RecursiveCharacterTextSplitter` while retaining all source metadata (`post_id`, `title`, `link`, `chunk_id`).
- **Local Embeddings**: Generates 768-dimensional vector embeddings with local **Ollama** (`nomic-embed-text`) or HuggingFace models.
- **Persistent Vector Store**: Uses **ChromaDB** on disk for zero-config persistence, metadata filtering, and high-performance similarity search.
- **Guardrailed Generation**: Powered by `llama3.2` with strict anti-hallucination prompts. If the answer is not present in the site content, it strictly responds:
  > *"I couldn’t find that in the site content"*
- **Interactive Terminal Interface**: Continuous CLI chat session for asking questions directly from your terminal.

---

## 📁 Project Structure

```text
e2m_rag_project1/
├── .env                     # Environment configuration (WordPress URL, Ollama models, Chroma settings)
├── requirements.txt         # Project dependencies
├── models/
│   ├── __init__.py
│   └── post.py              # Pydantic schemas (CleanedPost, EmbeddedChunk)
├── ingestion/
│   ├── __init__.py
│   ├── loader.py            # Fetches and cleans WordPress REST API content
│   └── splitter.py          # Splits content into chunks with preserved metadata
├── embedding/
│   ├── __init__.py
│   └── embedder.py          # Computes embeddings & persists vectors in ChromaDB
├── retrieval/
│   ├── __init__.py
│   └── retriever.py         # Top-k similarity retrieval from ChromaDB
├── generation/
│   ├── __init__.py
│   └── generator.py         # Guardrailed RAG chain with interactive CLI
└── chroma_db/               # Persistent on-disk vector database
```

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **`uv` Package Manager** (Ultra-fast Python package installer & resolver)
   - Install `uv` if not already installed:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
3. **Ollama**:
   - Download & install Ollama from [ollama.ai](https://ollama.ai).
   - Ensure the Ollama background daemon is running.
4. **Local WordPress Site** (e.g. Local by Flywheel at `http://e2m-project.local`).

---

## 🚀 Step-by-Step Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/kathanshah28/e2m_rag_project1.git
cd e2m_rag_project1
```

### 2. Create Virtual Environment with `uv`
```bash
uv venv ragenv
```

### 3. Activate the Environment
- **macOS / Linux**:
  ```bash
  source ragenv/bin/activate
  ```
- **Windows**:
  ```bash
  ragenv\Scripts\activate
  ```

### 4. Install Dependencies with `uv`
```bash
uv pip install -r requirements.txt
```

### 5. Setup Local Ollama Models
Pull the required embedding and LLM models locally:
```bash
# Pull 768-dimensional embedding model
ollama pull nomic-embed-text

# Pull the lightweight, high-performance LLM
ollama pull llama3.2
```

### 6. Configure Environment Variables (`.env`)
Create or verify your `.env` file with the following settings:

```dotenv
# WordPress Configuration
WORDPRESS_URL=http://e2m-project.local
WORDPRESS_USERNAME=
WORDPRESS_APP_PASSWORD=

# Ollama / LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Embeddings Configuration
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store & Storage
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=wordpress_rag
```

---

## 💻 Running the Pipeline

### Step 1: Ingest WordPress Posts
Fetches, cleans, and validates all posts from your WordPress REST API:
```bash
python ingestion/loader.py
```

### Step 2: Split Content into Chunks
Splits cleaned posts into chunks while preserving post IDs, titles, and links:
```bash
python ingestion/splitter.py
```

### Step 3: Embed & Store in ChromaDB
Computes embeddings using `nomic-embed-text` and stores them into persistent ChromaDB:
```bash
python embedding/embedder.py
```

### Step 4: Test Semantic Retrieval
Query ChromaDB directly to retrieve the top 5 most relevant document chunks:
```bash
python retrieval/retriever.py "What is the return and refund policy?"
```

### Step 5: Launch Interactive Q&A Assistant (CLI)
Start the interactive terminal assistant to chat with your WordPress knowledge base in real-time:
```bash
python generation/generator.py
```

You can also pass a direct query:
```bash
python generation/generator.py "How do I schedule maintenance for the UltraBlend 3000?"
```

### 🛠️ Debug Mode (`--debug`)
Run the generator with `--debug` (or `-d`) to inspect:
1. **Preprocessed / normalized query**
2. **Top-k retrieved chunks with similarity scores and titles**
3. **The exact rendered prompt sent to the LLM**

```bash
# Debug single query
python generation/generator.py --debug "What is the warranty period?"

# Or launch interactive chat in debug mode (or type :debug during chat)
python generation/generator.py --debug
```

---

## 🛡️ Anti-Hallucination Guardrail Demo

When asked about information **not** present in the WordPress site:
```text
Ask a question > What is the population of Tokyo?

Searching site content and generating response...
----------------------------------------------------------------------
ANSWER:
I couldn’t find that in the site content
----------------------------------------------------------------------
```

When asked about **known** WordPress site content:
```text
Ask a question > What is the warranty on the UltraBlend 3000?

Searching site content and generating response...
----------------------------------------------------------------------
ANSWER:
The UltraBlend 3000 comes with a 2-year manufacturer warranty covering motor and parts.

Sources:
  • UltraBlend 3000 Product Guide: http://e2m-project.local/ultrablend-3000-product-guide/
----------------------------------------------------------------------
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
