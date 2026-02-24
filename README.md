# Simple RAG

A minimal Retrieval-Augmented Generation (RAG) app. Upload documents and ask questions about them.

## Stack

- **Backend**: FastAPI, LangChain, ChromaDB, OpenAI
- **Frontend**: React, Vite, TailwindCSS

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
uvicorn main:app --reload
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Usage

1. Upload a PDF, TXT, or Markdown file via the sidebar
2. Ask questions in the chat — answers are grounded in your documents
3. Source file names are shown below each answer
