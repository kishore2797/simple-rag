import { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Upload,
  Send,
  FileText,
  Trash2,
  Bot,
  User,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";

const API = "/api";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("rag_messages") || "[]");
    } catch {
      return [];
    }
  });
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    axios.get(`${API}/documents`).then((res) => {
      if (res.data.documents?.length) {
        setDocuments(res.data.documents.map((name) => ({ name, chunks: null })));
      }
    }).catch(() => {});
  }, []);

  // useEffect(() => {
  //   localStorage.setItem("rag_messages", JSON.stringify(messages));
  // }, [messages]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setDocuments((prev) => [
        ...prev,
        { name: res.data.name, chunks: res.data.chunks },
      ]);
      setUploadStatus({ type: "success", message: `"${res.data.name}" uploaded — ${res.data.chunks} chunks indexed.` });
    } catch (err) {
      const msg = err.response?.data?.detail || "Upload failed.";
      setUploadStatus({ type: "error", message: msg });
    } finally {
      setUploading(false);
      fileInputRef.current.value = "";
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!question.trim() || querying) return;

    const userMsg = { role: "user", content: question };
    setMessages((prev) => [...prev, userMsg]);
    setQuestion("");
    setQuerying(true);

    setTimeout(scrollToBottom, 50);

    try {
      const res = await axios.post(`${API}/query`, { question: userMsg.content });
      const botMsg = {
        role: "assistant",
        content: res.data.answer,
        sources: res.data.sources,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const msg = err.response?.data?.detail || "Something went wrong.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${msg}`, sources: [] },
      ]);
    } finally {
      setQuerying(false);
      setTimeout(scrollToBottom, 50);
    }
  };

  const handleClear = async () => {
    try {
      await axios.delete(`${API}/documents`);
      setDocuments([]);
      setMessages([]);
      setUploadStatus(null);
      localStorage.removeItem("rag_messages");
    } catch {
      setUploadStatus({ type: "error", message: "Failed to clear documents." });
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
            <Bot size={18} />
          </div>
          <h1 className="text-lg font-semibold tracking-tight">Simple RAG</h1>
        </div>
        <span className="text-xs text-gray-500">Retrieval-Augmented Generation</span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 border-r border-gray-800 bg-gray-900 flex flex-col p-4 gap-4">
          <div>
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Documents
            </h2>

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
            >
              {uploading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Upload size={16} />
              )}
              {uploading ? "Uploading..." : "Upload Document"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt,.md"
              onChange={handleUpload}
              className="hidden"
            />
            <p className="text-xs text-gray-500 mt-1.5 text-center">
              PDF, TXT, or MD files
            </p>
          </div>

          {uploadStatus && (
            <div
              className={`flex items-start gap-2 rounded-lg p-3 text-xs ${
                uploadStatus.type === "success"
                  ? "bg-green-900/40 text-green-300"
                  : "bg-red-900/40 text-red-300"
              }`}
            >
              {uploadStatus.type === "success" ? (
                <CheckCircle2 size={14} className="mt-0.5 shrink-0" />
              ) : (
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
              )}
              {uploadStatus.message}
            </div>
          )}

          {documents.length > 0 && (
            <div className="flex-1 overflow-y-auto space-y-2">
              {documents.map((doc, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 p-2.5 rounded-lg bg-gray-800 text-xs"
                >
                  <FileText size={14} className="text-indigo-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="font-medium text-gray-200 break-all">{doc.name}</p>
                    <p className="text-gray-500">{doc.chunks} chunks</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {documents.length === 0 && (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-xs text-gray-600 text-center">
                No documents yet.<br />Upload one to get started.
              </p>
            </div>
          )}

          {documents.length > 0 && (
            <button
              onClick={handleClear}
              className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-gray-700 hover:bg-red-900/30 hover:border-red-700 hover:text-red-400 transition-colors text-sm text-gray-400"
            >
              <Trash2 size={14} />
              Clear All
            </button>
          )}
        </aside>

        {/* Chat area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center gap-3">
                <div className="w-16 h-16 rounded-2xl bg-gray-800 flex items-center justify-center">
                  <Bot size={32} className="text-indigo-400" />
                </div>
                <h2 className="text-xl font-semibold text-gray-300">Ask your documents</h2>
                <p className="text-sm text-gray-500 max-w-sm">
                  Upload a PDF, TXT, or Markdown file on the left, then ask any question about its content.
                </p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {msg.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 mt-1">
                    <Bot size={16} />
                  </div>
                )}

                <div className={`max-w-2xl ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col gap-1`}>
                  <div
                    className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-indigo-600 text-white rounded-tr-sm"
                        : "bg-gray-800 text-gray-100 rounded-tl-sm"
                    }`}
                  >
                    {msg.content}
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="flex flex-wrap gap-1 px-1">
                      {msg.sources.map((src, j) => (
                        <span
                          key={j}
                          className="text-xs bg-gray-800 text-indigo-400 px-2 py-0.5 rounded-full border border-gray-700"
                        >
                          {src}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0 mt-1">
                    <User size={16} />
                  </div>
                )}
              </div>
            ))}

            {querying && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0">
                  <Bot size={16} />
                </div>
                <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-2">
                  <Loader2 size={14} className="animate-spin text-indigo-400" />
                  <span className="text-sm text-gray-400">Thinking...</span>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-800 bg-gray-900 px-6 py-4">
            <form onSubmit={handleQuery} className="flex gap-3">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={
                  documents.length === 0
                    ? "Upload a document first..."
                    : "Ask a question about your documents..."
                }
                disabled={documents.length === 0 || querying}
                className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm placeholder-gray-500 focus:outline-none focus:border-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              />
              <button
                type="submit"
                disabled={!question.trim() || querying || documents.length === 0}
                className="px-4 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                <Send size={16} />
              </button>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}
