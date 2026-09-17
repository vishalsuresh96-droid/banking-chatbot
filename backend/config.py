import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = PROJECT_ROOT / "data" / "banking_knowledge_base_1000.csv"
INDEX_DIR = PROJECT_ROOT / "storage" / "faiss_index"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TOP_K = int(os.getenv("TOP_K", "3"))
# Keep the server usable in restricted/offline environments.  Set this to true
# only when the embedding model has not been cached and downloads are allowed.
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "false").lower() == "true"
# Transformer initialization can still be slow or blocked on locked-down networks.
# The built-in TF-IDF retriever is the reliable default for this standalone app.
ENABLE_SEMANTIC_RETRIEVAL = os.getenv("ENABLE_SEMANTIC_RETRIEVAL", "false").lower() == "true"
