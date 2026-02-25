import re
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
from langchain.prompts import PromptTemplate
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

PROMPT_TEMPLATE = """Use the context below to answer the question. Give a short, direct answer only.

Important:
- Any section starting with [FRONT MATTER] contains the book's title page and copyright page. Use it as the definitive source for the book's title, editor, publisher, and year.
- Do not mention other books referenced or cited inside the text.
- Do not say "not explicitly stated" or "however". Just answer.
- If the context truly does not contain the answer, say only: "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer (one or two sentences, no preamble):"""

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

    retrieved = vectorstore.similarity_search(request.question, k=8)

    front_matter_docs = [
        doc for doc, meta in zip(collection["documents"], collection["metadatas"])
        if meta and meta.get("chunk_type") == "front_matter"
    ]
    seen_ids = {doc.page_content[:100] for doc in retrieved}
    for fm_text in front_matter_docs:
        if fm_text[:100] not in seen_ids:
            retrieved.insert(0, Document(page_content=fm_text, metadata={"chunk_type": "front_matter"}))
            seen_ids.add(fm_text[:100])

    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved)
    filled_prompt = prompt.format(context=context, question=request.question)
    raw_answer = llm.invoke(filled_prompt).content

    result = {"result": raw_answer, "source_documents": retrieved}

    answer = result["result"]
    answer = re.sub(
        r"^[\w\s',]+(?:is not explicitly stated|is not provided|cannot be determined|is unclear)"
        r"[\w\s',\.]*?(?:However,|But,|That said,)?\s*"
        r"(?:according to (?:the )?\[FRONT MATTER\],?\s*|based on (?:the )?(?:provided )?context,?\s*|the (?:provided )?context (?:states?|indicates?|shows?|reveals?) that\s*)?",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    answer = re.sub(
        r"^(?:According to|Based on)(?: the)?(?: \[FRONT MATTER\]| provided context| context),?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    if answer:
        answer = answer[0].upper() + answer[1:]

    sources = list(
        {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
    )

    return QueryResponse(answer=answer, sources=sources)


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
