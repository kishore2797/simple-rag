# Simple RAG

A local, privacy-first Retrieval-Augmented Generation (RAG) app. Upload documents and ask questions about them — everything runs on your machine with no API keys required.

---

## What Type of RAG Is This?

This project implements **Naive RAG** — the foundational RAG pattern.

### RAG Type: Naive / Basic RAG

| Property | Value |
|---|---|
| **Category** | Naive RAG (Chunk-based, Dense, Single-hop, Local) |
| **Retrieval** | Dense vector search (cosine similarity) |
| **Chunking** | Fixed-size with character overlap |
| **Embedding** | Local model via Ollama (`nomic-embed-text`) |
| **LLM** | Local model via Ollama (`llama3.2`) |
| **Vector Store** | ChromaDB (local SQLite) |
| **Enhancements** | Front-matter injection, MMR-ready, metadata-aware prompt |

### How it fits in the RAG landscape

```
Naive RAG  →  Advanced RAG  →  Modular RAG  →  Agentic RAG
   ↑
You are here
```

**Naive RAG** is the starting point of all RAG systems:
1. Load documents
2. Split into chunks
3. Embed chunks → store in vector DB
4. At query time: embed question → retrieve top-k chunks → generate answer

---

## Types of RAG (and where this project fits)

### By Retrieval Strategy

| Type | Description | Used Here |
|---|---|---|
| **Naive RAG** | Basic retrieve-then-generate | ✅ Yes |
| **Advanced RAG** | Adds query rewriting, re-ranking, hybrid search | Partial (front-matter injection) |
| **Modular RAG** | Swappable retriever/reranker/generator modules | ❌ No |

### By Retrieval Method

| Type | Description | Used Here |
|---|---|---|
| **Dense RAG** | Cosine similarity over embeddings | ✅ Yes |
| **Sparse RAG** | BM25 / keyword matching | ❌ No |
| **Hybrid RAG** | Dense + sparse combined | ❌ No |

### By Architecture

| Type | Description | Used Here |
|---|---|---|
| **Single-hop RAG** | One retrieval step per query | ✅ Yes |
| **Multi-hop RAG** | Iterative retrieval for complex reasoning | ❌ No |
| **Agentic RAG** | Agent decides when/what to retrieve | ❌ No |
| **Self-RAG** | Model critiques its own retrieval | ❌ No |
| **Corrective RAG** | Falls back to web search if retrieval quality is low | ❌ No |
| **Graph RAG** | Retrieves from a knowledge graph | ❌ No |

### By Knowledge Source

| Type | Description | Used Here |
|---|---|---|
| **Local RAG** | Private files on your machine | ✅ Yes |
| **Web RAG** | Real-time web retrieval | ❌ No |
| **Structured RAG** | Databases/tables via Text-to-SQL | ❌ No |
| **Multimodal RAG** | Images, audio, video alongside text | ❌ No |

---

## All 33 RAG Types — Complete Reference

### 1. By Retrieval Strategy

| # | Type | Description |
|---|---|---|
| 1 | **Naive RAG** | Basic retrieve-then-generate; chunk → embed → retrieve → generate |
| 2 | **Advanced RAG** | Adds pre/post retrieval steps: query rewriting, re-ranking, hybrid search |
| 3 | **Modular RAG** | Composable modules (retriever, reranker, generator) swapped independently |

### 2. By Retrieval Granularity

| # | Type | Description |
|---|---|---|
| 4 | **Chunk-based RAG** | Fixed or semantic text chunks (most common) |
| 5 | **Sentence-level RAG** | Finer granularity — splits and retrieves at sentence level |
| 6 | **Document-level RAG** | Retrieves entire documents rather than chunks |

### 3. By Index / Search Type

| # | Type | Description |
|---|---|---|
| 7 | **Dense RAG** | Vector similarity search using embedding models |
| 8 | **Sparse RAG** | Keyword-based retrieval: BM25, TF-IDF |
| 9 | **Hybrid RAG** | Combines dense + sparse with fusion (RRF or linear) |

### 4. By Architecture

| # | Type | Description |
|---|---|---|
| 10 | **Single-hop RAG** | One retrieval step per query |
| 11 | **Multi-hop RAG** | Iterative retrieval; each step informs the next for complex reasoning |
| 12 | **Agentic RAG** | LLM agent decides when and what to retrieve dynamically |
| 13 | **Self-RAG** | Model learns to critique and selectively retrieve using special tokens |
| 14 | **Corrective RAG (CRAG)** | Evaluates retrieved docs; falls back to web search if quality is low |
| 15 | **Graph RAG** | Retrieves from a knowledge graph instead of flat vector store |

### 5. By Knowledge Source

| # | Type | Description |
|---|---|---|
| 16 | **Local RAG** | Private/local document stores — no external APIs |
| 17 | **Web RAG** | Real-time web retrieval for up-to-date information |
| 18 | **Structured RAG** | Retrieves from databases/tables via Text-to-SQL + RAG |
| 19 | **Multimodal RAG** | Retrieves images, audio, video alongside text |

### 6. Known & Verifiable Variants

| # | Type | Description |
|---|---|---|
| 20 | **Hybrid Multilingual RAG** | Cross-lingual retrieval; queries in one language retrieve docs in another (mBERT, LaBSE) |
| 21 | **HiRAG / HiFi-RAG** | Hierarchical RAG; chunks at multiple granularity levels (sentence → paragraph → section) |
| 22 | **Bidirectional RAG** | Retrieval forward (query→docs) and backward (docs→query verification) for answer validation |
| 23 | **Graph-RAG / Graph-01** | Microsoft's GraphRAG; graph traversal for multi-hop reasoning |
| 24 | **Mega-RAG** | Scales RAG to millions of docs using ANN indexes like FAISS/ScaNN at production scale |

### 7. Advanced Architecture Variants

| # | Type | Description |
|---|---|---|
| 25 | **Classic RAG** | The original baseline: vector embed → similarity search → LLM |
| 26 | **Mindscape-Aware RAG** | Personalised retrieval using user mental models, history, and behavioral patterns |
| 27 | **Hypergraph Memory RAG** | Multi-entity relational retrieval using hypergraphs (one relation connects multiple nodes) |
| 28 | **QuCo-RAG** | Query-Context Optimization: intelligent query rewriting + context compression |
| 29 | **Affordance RAG** | Retrieves not just knowledge but actionable tools/APIs — knowledge + tool selection |
| 30 | **SignRAG** | Retrieval for sign-language/gesture systems; inputs are sign video and motion capture |
| 31 | **TV-RAG** | Temporal-Visual RAG; time-aware video segment retrieval and reasoning |
| 32 | **RAGPart** | Partial Context RAG — passes only the highest-scoring segments from retrieved docs |
| 33 | **RAGMask** | Token-level masking of irrelevant content before LLM processing |

---

## When to Use This Type of RAG

### ✅ Good fit for:
- Asking questions over **personal documents** (books, reports, manuals)
- **Private/offline** use cases where data must not leave the machine
- **Prototyping** a RAG system before adding complexity
- Single-topic corpora where context is self-contained
- Learning and understanding how RAG works fundamentally

### ❌ Not suited for:
- Questions requiring **cross-document reasoning** (use Multi-hop RAG)
- **Real-time information** needs (use Web RAG)
- **Structured data** like spreadsheets or databases (use Text-to-SQL + RAG)
- Large corpora with millions of documents (use Mega-RAG / production vector DBs)
- When retrieval accuracy must be near-perfect (add re-ranking → Advanced RAG)

---

## The 5 Steps of This RAG Pipeline

```
1. INGEST        Upload PDF/TXT/MD file
                      ↓
2. CHUNK         Split into 700-char chunks, 150-char overlap
                      ↓
3. EMBED         nomic-embed-text converts each chunk → vector [768 dims]
                      ↓
4. STORE         Vectors + text saved in ChromaDB (local SQLite)
                      ↓
   [At query time]
                      ↓
5. RETRIEVE      Question embedded → cosine similarity search → top 8 chunks
      +           Front-matter chunk always injected (for metadata accuracy)
   GENERATE       llama3.2 reads chunks + answers the question
```

---

## Enhancements Over Basic Naive RAG

This project adds several improvements on top of the baseline:

| Enhancement | What it does | Why |
|---|---|---|
| **Front-matter injection** | Always includes title/copyright page in context | Prevents hallucinating wrong author/year |
| **Publication year annotation** | Extracts and labels the correct year from front matter | Avoids LoC control number confusion |
| **Metadata-aware prompt** | Instructs LLM to trust `[FRONT MATTER]` over inline citations | Reduces citation confusion |
| **Smaller chunks (700 chars)** | Finer granularity than default 1000 chars | Better isolation of specific facts |
| **`k=8` retrieval** | Fetches more candidates than default `k=4` | Improves recall for conceptual passages |
| **MMR-ready** | Can switch to `search_type="mmr"` in one line | Prevents retrieving 8 near-identical chunks |

---

## Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | FastAPI + Uvicorn | REST API server |
| **LLM** | Ollama + `llama3.2` | Local text generation |
| **Embeddings** | Ollama + `nomic-embed-text` | Local vector encoding |
| **Orchestration** | LangChain | Document loading, splitting, prompting |
| **Vector DB** | ChromaDB (SQLite) | Persistent local vector storage |
| **Document parsing** | PyPDF, TextLoader | PDF and text file ingestion |
| **Frontend** | React + Vite + TailwindCSS | Chat UI |

---

## Setup

### Prerequisites
- [Ollama](https://ollama.com) installed
- Python 3.10+
- Node.js 18+

### 1. Pull local models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Usage

1. **Upload** a PDF, TXT, or Markdown file via the sidebar
2. Wait for indexing (embedding ~800–1200 chunks for a 400-page PDF takes 30–90s)
3. **Ask questions** in the chat — answers are grounded in your documents
4. Source file names are shown below each answer
5. **Clear All** removes all documents and resets the vector store

---

## Limitations (Naive RAG)

| Limitation | Description | Fix (if needed) |
|---|---|---|
| No query rewriting | Question is used as-is for retrieval | Add MultiQueryRetriever or HyDE |
| No re-ranking | Top-k by cosine only, no cross-encoder | Add `cross-encoder/ms-marco-MiniLM` |
| No hybrid search | Vector only, no BM25 | Add `BM25Retriever` + `EnsembleRetriever` |
| No deduplication | Uploading same file twice duplicates chunks | Add file hash check before indexing |
| Fixed chunk size | Doesn't respect sentence/paragraph boundaries | Switch to `SemanticChunker` |
| Single-hop only | Can't reason across multiple documents | Upgrade to Agentic or Multi-hop RAG |

---

## Roadmap (Next Enhancements)

- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Cross-encoder re-ranking
- [ ] Duplicate file detection
- [ ] Semantic chunking
- [ ] Chat history with memory
- [ ] Multi-document comparison queries
