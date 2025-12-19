import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------
# FORCE FULL OFFLINE MODE
# ----------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(
    os.getenv("HF_HOME", "/hf"),
    "hub"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentimentVectorStore:
    def __init__(self):
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )

        self.collection = self.client.get_or_create_collection(
            name="sentiment_analysis"
        )

        # ✅ FINAL FIX: correct cache + offline
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=os.path.join(os.getenv("HF_HOME", "/hf"), "hub"),
            local_files_only=True
        )

    def add_record(self, text, sentiment, confidence):
        embedding = self.embedder.encode(text).tolist()

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "sentiment": sentiment,
                "confidence": confidence
            }],
            ids=[str(uuid.uuid4())]
        )

    def similarity_search(self, query, top_k=3):
        query_embedding = self.embedder.encode(query).tolist()

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
