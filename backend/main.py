import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema import Document

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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
)

PROMPT_TEMPLATE = """You are an assistant that answers questions strictly based on the provided context.

Rules:
- Use ONLY the context below to answer. Do not use outside knowledge.
- Chunks marked [FRONT MATTER] contain the book's own title page, copyright page, and publication info. These are the authoritative source for: title, editor, author, publisher, ISBN, and publication year. Always prefer [FRONT MATTER] over any citations or references found elsewhere in the text.
- Do NOT confuse books cited or referenced within the text with the book being asked about.
- If the answer is not found in the context, say "I don't have enough information to answer that."
- Be specific and direct.

Context:
{context}

Question: {question}

Answer:"""

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
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = file.filename
        chunk.metadata["chunk_type"] = "body"

    extra_chunks = []
    if suffix == ".pdf" and documents:
        front_pages = documents[:min(5, len(documents))]
        front_text = "[FRONT MATTER]\n" + "\n".join(p.page_content for p in front_pages)
        extra_chunks.append(Document(
            page_content=front_text,
            metadata={"source": file.filename, "chunk_type": "front_matter"},
        ))

    all_chunks = extra_chunks + chunks
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

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa_chain.invoke({"query": request.question})

    sources = list(
        {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
    )

    return QueryResponse(answer=result["result"], sources=sources)


@app.delete("/documents")
async def clear_documents():
    global vectorstore
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
