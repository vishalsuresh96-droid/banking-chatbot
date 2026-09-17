"""A small, explicit RAG pipeline: CSV -> documents -> embeddings -> FAISS -> answer."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from backend.config import (
    ALLOW_MODEL_DOWNLOAD,
    EMBEDDING_MODEL,
    ENABLE_SEMANTIC_RETRIEVAL,
    INDEX_DIR,
    OPENAI_MODEL,
    TOP_K,
)


class RAGError(Exception):
    """Raised when the local RAG pipeline cannot complete a request."""


@dataclass
class Document:
    """One FAQ row, kept with useful metadata for displaying sources."""

    section: str
    question: str
    answer: str
    content: str


def load_documents(dataset_path: Path) -> list[Document]:
    """Read and validate the supplied FAQ CSV without assuming extra columns."""
    if not dataset_path.exists():
        raise RAGError(f"Dataset not found: {dataset_path}")

    try:
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required_columns = {"Section", "Question", "Answer"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise RAGError("Dataset must contain Section, Question, and Answer columns.")

            documents = []
            for row_number, row in enumerate(reader, start=2):
                section = (row.get("Section") or "").strip()
                question = (row.get("Question") or "").strip()
                answer = (row.get("Answer") or "").strip()
                if not all((section, question, answer)):
                    raise RAGError(f"Dataset row {row_number} has an empty required value.")
                content = f"Section: {section}\nQuestion: {question}\nAnswer: {answer}"
                documents.append(Document(section, question, answer, content))
    except UnicodeDecodeError as error:
        raise RAGError("Could not read the dataset as UTF-8 CSV.") from error
    except OSError as error:
        raise RAGError(f"Could not read dataset: {error}") from error

    if not documents:
        raise RAGError("The dataset has no FAQ records.")
    return documents


class BankingRAG:
    """Builds or loads a local FAISS index, retrieves FAQ rows, and calls an LLM."""

    def __init__(self, dataset_path: Path, index_dir: Path = INDEX_DIR) -> None:
        self.dataset_path = dataset_path
        self.index_dir = index_dir
        self.documents = load_documents(dataset_path)
        self.embedder = None
        self.index = None
        self.retrieval_mode = "keyword"
        if ENABLE_SEMANTIC_RETRIEVAL:
            try:
                # Import lazily: the default keyword mode must have a fast,
                # dependency-light startup path.
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer(
                    EMBEDDING_MODEL, local_files_only=not ALLOW_MODEL_DOWNLOAD
                )
                self.index = self._load_or_create_index()
                self.retrieval_mode = "semantic"
            except Exception:
                self._build_keyword_index()
        else:
            # A browser chat should not fail just because a large optional model
            # is not installed or a corporate network blocks model downloads.
            self._build_keyword_index()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Normalize words consistently for the local, dependency-free fallback."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_keyword_index(self) -> None:
        """Build a small TF-IDF index used whenever semantic retrieval is unavailable."""
        term_counts = []
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            # Questions are the strongest signal for FAQ matching.
            terms = self._tokens(f"{document.question} {document.question} {document.question} {document.section} {document.answer}")
            counts = Counter(terms)
            term_counts.append(counts)
            document_frequency.update(counts.keys())

        total = len(self.documents)
        self._keyword_vectors = []
        for counts in term_counts:
            vector = {
                term: (1 + math.log(count)) * (math.log((1 + total) / (1 + document_frequency[term])) + 1)
                for term, count in counts.items()
            }
            norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
            self._keyword_vectors.append((vector, norm))
        self._keyword_idf = {
            term: math.log((1 + total) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }

    @property
    def _index_path(self) -> Path:
        return self.index_dir / "banking_faq.index"

    @property
    def _metadata_path(self) -> Path:
        return self.index_dir / "documents.json"

    def _load_or_create_index(self) -> faiss.Index:
        """Reuse the saved index only when it matches the current document count."""
        if self._index_path.exists() and self._metadata_path.exists():
            try:
                saved_documents = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                index = faiss.read_index(str(self._index_path))
                if len(saved_documents) == len(self.documents) == index.ntotal:
                    self.documents = [Document(**item) for item in saved_documents]
                    return index
            except (OSError, ValueError, TypeError, RuntimeError):
                pass  # A corrupt or old index is safely rebuilt below.

        return self._create_and_save_index()

    def _create_and_save_index(self) -> faiss.Index:
        try:
            texts = [document.content for document in self.documents]
            vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            vectors = np.asarray(vectors, dtype="float32")
            index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity after normalization
            index.add(vectors)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self._index_path))
            self._metadata_path.write_text(
                json.dumps([asdict(document) for document in self.documents], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return index
        except Exception as error:
            raise RAGError(f"Could not create the FAISS index: {error}") from error

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Return the most similar FAQ records for a non-empty user question."""
        question = question.strip()
        if not question:
            raise RAGError("Please enter a question.")
        if self.embedder is None or self.index is None:
            return self._keyword_retrieve(question, top_k)
        try:
            query_vector = self.embedder.encode([question], normalize_embeddings=True)
            scores, ids = self.index.search(np.asarray(query_vector, dtype="float32"), min(top_k, len(self.documents)))
        except Exception as error:
            raise RAGError(f"Could not search the knowledge base: {error}") from error

        return [
            {"section": self.documents[item_id].section, "question": self.documents[item_id].question,
             "answer": self.documents[item_id].answer, "score": round(float(score), 3)}
            for score, item_id in zip(scores[0], ids[0]) if item_id != -1
        ]

    def _keyword_retrieve(self, question: str, top_k: int) -> list[dict]:
        """Return the most relevant local FAQ rows without network or ML-model access."""
        query_counts = Counter(self._tokens(question))
        if not query_counts:
            raise RAGError("Please enter a question containing letters or numbers.")
        query = {
            term: (1 + math.log(count)) * self._keyword_idf.get(term, 0.0)
            for term, count in query_counts.items() if term in self._keyword_idf
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query.values()))
        if not query_norm:
            return []

        ranked = []
        for item_id, (vector, document_norm) in enumerate(self._keyword_vectors):
            score = sum(weight * vector.get(term, 0.0) for term, weight in query.items())
            ranked.append((score / (query_norm * document_norm), item_id))
        ranked.sort(reverse=True)
        return [
            {
                "section": self.documents[item_id].section,
                "question": self.documents[item_id].question,
                "answer": self.documents[item_id].answer,
                "score": round(float(score), 3),
            }
            for score, item_id in ranked[: min(top_k, len(ranked))]
            if score > 0
        ]

    def answer(self, question: str) -> dict:
        """Generate an answer constrained to the FAISS-retrieved FAQ context."""
        sources = self.retrieve(question)
        context = "\n\n".join(
            f"FAQ {number}\nSection: {source['section']}\nQuestion: {source['question']}\nAnswer: {source['answer']}"
            for number, source in enumerate(sources, start=1)
        )
        try:
            client = OpenAI()
            response = client.responses.create(
                model=OPENAI_MODEL,
                instructions=(
                    "You are a Banking FAQ assistant. Answer using only the FAQ context provided. "
                    "Do not add facts, advice, figures, policies, or assumptions not present in that context. "
                    "If the context does not answer the question, say exactly: "
                    "I could not find that information in the banking FAQ knowledge base."
                ),
                input=f"User question: {question.strip()}\n\nFAQ context:\n{context}",
                max_output_tokens=300,
            )
            answer = (response.output_text or "").strip()
            if not answer:
                raise RAGError("The language model returned an empty response.")
            return {"answer": answer, "sources": sources}
        except RAGError:
            raise
        except Exception as error:
            raise RAGError(
                "Could not generate an answer. Check OPENAI_API_KEY, OPENAI_MODEL, and your internet connection."
            ) from error
