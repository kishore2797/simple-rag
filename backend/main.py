import re
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from flashrank import Ranker, RerankRequest

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

app = FastAPI(title="Simple RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="documents",
)

# Child splitter — small chunks for precise retrieval
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
)

# Parent splitter — larger chunks sent to LLM for full context
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
)

# Reranker (runs fully locally, no API needed)
reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(BASE_DIR / "reranker_cache"))

# In-memory store: child_chunk_id → parent Document
parent_store: dict[str, Document] = {}

PROMPT_TEMPLATE = """You are answering questions strictly based on the provided context. Do NOT use your training knowledge.

Rules:
- Answer ONLY using the context below. If the context contains an exact quote or specific formulation, use that — do not paraphrase into a generic version.
- Any section starting with [FRONT MATTER] is the definitive source for title, editor, publisher, and year.
- Do not reference other books cited in the text.
- Do not say "not explicitly stated", "however", or "I believe". Just answer directly.
- If the context does not contain the answer, say only: "I don't have enough information to answer that."
- When asked about a prediction or claim, quote the exact wording from the context, not a popular paraphrase.

Context:
{context}

Question: {question}

Answer (one or two sentences using the exact wording from context where possible):"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,   
    input_variables=["context", "question"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class DocumentInfo(BaseModel):
    name: str
    chunks: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Simple RAG API is running"}


@app.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    allowed_types = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed_types)}",
        )

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))    
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")

    documents = loader.load()

    # --- UPGRADE 3: Parent-Child Chunking ---
    # Split into large parent chunks (1200 chars) for rich LLM context
    parents = parent_splitter.split_documents(documents)
    for i, p in enumerate(parents):
        p.metadata["source"] = file.filename
        p.metadata["chunk_type"] = "parent"
        p.metadata["parent_id"] = f"{file.filename}::parent::{i}"

    # Split into small child chunks (400 chars) for precise retrieval
    child_chunks = []
    for i, parent in enumerate(parents):
        children = child_splitter.split_text(parent.page_content)
        for j, child_text in enumerate(children):
            child_id = f"{file.filename}::parent::{i}::child::{j}"
            child_doc = Document(
                page_content=child_text,
                metadata={
                    "source": file.filename,
                    "chunk_type": "child",
                    "parent_id": f"{file.filename}::parent::{i}",
                    "child_id": child_id,
                },
            )
            child_chunks.append(child_doc)
            parent_store[child_id] = parent  # map child → parent

    # --- Front Matter (metadata chunk for title/editor/year) ---
    extra_chunks = []
    if suffix == ".pdf" and documents:
        front_pages = documents[:min(5, len(documents))]
        raw_front = "\n".join(p.page_content for p in front_pages)

        pub_year = None
        copyright_match = re.search(r"©.*?(\d{4})|Springer.*?(\d{4})|published.*?(\d{4})", raw_front, re.IGNORECASE)
        if copyright_match:
            pub_year = next(y for y in copyright_match.groups() if y)

        annotation = ""
        if pub_year:
            annotation = f"[PUBLICATION YEAR: {pub_year}] [NOTE: Any other years on this page (e.g. Library of Congress control numbers, ISBN registration years) are NOT the publication year.]\n\n"

        front_text = "[FRONT MATTER]\n" + annotation + raw_front
        extra_chunks.append(Document(
            page_content=front_text,
            metadata={"source": file.filename, "chunk_type": "front_matter"},
        ))

    all_chunks = extra_chunks + child_chunks
    vectorstore.add_documents(all_chunks)

    return DocumentInfo(name=file.filename, chunks=len(all_chunks))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    collection = vectorstore.get()
    if not collection["ids"]:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload a document first.",
        )

    # --- UPGRADE 1: Hybrid Retrieval (Dense + BM25) with RRF Fusion ---
    # Step 1a: Dense retrieval (semantic similarity) — two passes:
    # Pass 1: literal question
    # Pass 2: rephrase question to catch specific formulations (avoids semantic drift)
    dense_results = vectorstore.similarity_search(request.question, k=10)
    # Second pass with expanded phrasing to pull specific book quotes
    rephrased = request.question + " exact quote specific formulation stated in the text"
    dense_results2 = vectorstore.similarity_search(rephrased, k=6)
    # Merge both passes, keeping order (pass1 first for RRF ranking)
    seen_pass = {d.page_content[:120] for d in dense_results}
    for d in dense_results2:
        if d.page_content[:120] not in seen_pass:
            dense_results.append(d)
            seen_pass.add(d.page_content[:120])
    dense_body = [d for d in dense_results if d.metadata.get("chunk_type") != "front_matter"]

    # Step 1b: BM25 retrieval (keyword matching) over all stored body chunks
    all_body_texts = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(collection["documents"], collection["metadatas"])
        if meta and meta.get("chunk_type") == "child"
    ]
    bm25_results = []
    if all_body_texts:
        bm25_retriever = BM25Retriever.from_documents(all_body_texts)
        bm25_retriever.k = 12
        bm25_results = bm25_retriever.invoke(request.question)

    # Step 1c: RRF (Reciprocal Rank Fusion) — merge dense + BM25 rankings
    rrf_scores: dict[str, float] = {}
    rrf_docs: dict[str, Document] = {}
    k_rrf = 60  # standard RRF constant
    for rank, doc in enumerate(dense_body):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k_rrf + rank + 1)
        rrf_docs[key] = doc
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k_rrf + rank + 1)
        rrf_docs[key] = doc

    fused = sorted(rrf_docs.values(), key=lambda d: rrf_scores[d.page_content[:120]], reverse=True)[:12]

    # --- UPGRADE 3: Replace child chunks with their parent chunks ---
    # Child chunks matched semantically; return full parent context to LLM
    context_docs: list[Document] = []
    seen_parents: set[str] = set()
    for doc in fused:
        pid = doc.metadata.get("parent_id") or doc.metadata.get("child_id", "")
        parent = parent_store.get(doc.metadata.get("child_id", ""))
        if parent and pid not in seen_parents:
            context_docs.append(parent)
            seen_parents.add(pid)
        elif pid not in seen_parents:
            context_docs.append(doc)
            seen_parents.add(pid)

    # --- Keyword injection: catch exact-quote chunks that semantic search misses ---
    # Extract key nouns/terms from question (words >4 chars, ignore stopwords)
    _stopwords = {"what", "which", "when", "where", "does", "make", "made", "have",
                  "that", "this", "from", "with", "about", "were", "will", "would",
                  "their", "there", "these", "those", "into", "over", "also", "some"}
    question_keywords = [
        w.lower() for w in re.findall(r"[a-zA-Z]+", request.question)
        if len(w) > 4 and w.lower() not in _stopwords
    ]
    if question_keywords:
        seen_context = {d.page_content[:120] for d in context_docs}
        for doc_text, meta in zip(collection["documents"], collection["metadatas"]):
            if not meta or meta.get("chunk_type") != "child":
                continue
            doc_lower = doc_text.lower()
            # inject if at least 2 keywords appear in the chunk
            match_count = sum(1 for kw in question_keywords if kw in doc_lower)
            if match_count >= 2 and doc_text[:120] not in seen_context:
                injected = Document(page_content=doc_text, metadata=meta)
                parent = parent_store.get(meta.get("child_id", ""))
                if parent and parent.page_content[:120] not in seen_context:
                    context_docs.append(parent)
                    seen_context.add(parent.page_content[:120])
                elif doc_text[:120] not in seen_context:
                    context_docs.append(injected)
                    seen_context.add(doc_text[:120])

    # --- UPGRADE 2: Re-ranking with cross-encoder ---
    # Cross-encoder scores each (question, chunk) pair jointly — much more accurate than cosine
    if context_docs:
        rerank_request = RerankRequest(
            query=request.question,
            passages=[{"id": i, "text": d.page_content} for i, d in enumerate(context_docs)],
        )
        rerank_results = reranker.rerank(rerank_request)
        reranked_indices = [r["id"] for r in sorted(rerank_results, key=lambda x: x["score"], reverse=True)[:6]]
        context_docs = [context_docs[i] for i in reranked_indices]

    # Always inject front-matter chunk first for metadata questions
    front_matter_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(collection["documents"], collection["metadatas"])
        if meta and meta.get("chunk_type") == "front_matter"
    ]
    seen_fm = {d.page_content[:100] for d in context_docs}
    for fm_doc in front_matter_docs:
        if fm_doc.page_content[:100] not in seen_fm:
            context_docs.insert(0, fm_doc)
            seen_fm.add(fm_doc.page_content[:100])

    context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    filled_prompt = prompt.format(context=context, question=request.question)
    raw_answer = llm.invoke(filled_prompt).content

    result = {"result": raw_answer, "source_documents": context_docs}

    answer = result["result"].strip()

    sources = list(
        {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
    )

    return QueryResponse(answer=answer, sources=sources)


@app.delete("/documents")
async def clear_documents():
    global vectorstore, parent_store
    parent_store.clear()
    vectorstore.delete_collection()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(exist_ok=True)

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="documents",
    )

    return {"message": "All documents cleared successfully"}


@app.get("/documents")
async def list_documents():
    collection = vectorstore.get()
    if not collection["ids"]:
        return {"documents": [], "total_chunks": 0}

    sources = set()
    for meta in collection["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])

    return {"documents": list(sources), "total_chunks": len(collection["ids"])}
