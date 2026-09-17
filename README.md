# Banking FAQ Chatbot using LLM and RAG

A small beginner-friendly Retrieval-Augmented Generation (RAG) chatbot built from the supplied banking FAQ CSV. The chatbot searches the local FAQ knowledge base with FAISS, then asks an LLM to answer only from the FAQs it retrieved.

## What this project does

1. Reads `data/banking_knowledge_base_1000.csv`.
2. Turns each `Section`, `Question`, and `Answer` row into one document.
3. Creates embeddings with `all-MiniLM-L6-v2`.
4. Saves a FAISS vector index locally in `storage/faiss_index/`.
5. Retrieves the three closest FAQ documents for each question.
6. Sends only those documents and the question to the LLM.
7. Shows the grounded response and the retrieved FAQ sources in a browser chat UI.

## Project layout

```text
backend/              FastAPI server and RAG pipeline
frontend/             HTML, CSS, and JavaScript chat interface
data/                 Supplied banking FAQ dataset
storage/faiss_index/  Automatically generated local FAISS files
```

## Installation

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

If `python` opens the Microsoft Store or is unavailable on Windows, use the installed interpreter's full path to create the environment, for example:

```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
```

Open `.env` and replace `your_openai_api_key_here` with your API key. Keep `.env` private and do not commit it.

## Run

```powershell
uvicorn backend.app:app --reload
```

Visit http://127.0.0.1:8000 in your browser.

The default launch uses a local TF-IDF retriever, so the frontend and API start reliably offline. If you have the embedding model locally, set `ENABLE_SEMANTIC_RETRIEVAL=true` for FAISS retrieval. For an initial model download, additionally set `ALLOW_MODEL_DOWNLOAD=true`. Delete `storage/faiss_index/` if you replace or edit the dataset, then restart the app.

## API

`POST /chat`

```json
{ "question": "What is KYC?" }
```

`GET /health` returns the current server status and number of loaded documents.

## Error handling

The app reports clear errors for missing dataset columns or data, an unavailable embedding model, corrupt indexes, blank questions, missing OpenAI configuration, and LLM/API failures.

## Important limitation

This is an educational project, not financial advice. The LLM is instructed to answer only from the retrieved FAQ records. Retrieval quality still affects the response, so confirm important banking information with the relevant bank or official source.
