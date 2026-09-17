from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from backend.rag_pipeline import BankingRAG
from backend.config import DATASET_PATH

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading BankingRAG FAISS index for fast retrieval...")
try:
    rag = BankingRAG(DATASET_PATH)
    print("LOADED FAQs")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    rag = None

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline is not available.")
    return {
        "status": "ok",
        "documents": len(rag.documents),
        "retrieval_mode": rag.retrieval_mode,
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized properly.")
    try:
        sources = rag.retrieve(req.question, top_k=3)
        if sources:
            # Using the top retrieved answer directly avoids slow API calls
            return {"answer": sources[0]["answer"], "sources": sources}
        return {"answer": "Sorry, try asking like: What is KYC? What is FD?", "sources": []}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def home():
    return FileResponse(frontend_path / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
