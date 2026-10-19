# Dense Embeddings RAG method
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_retriever(documents, dataset_name):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode_document(
        [document["text"] for document in documents],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    def retrieve(query, top_k):
        scores = np.dot(embeddings, model.encode_query(query, normalize_embeddings=True))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"id": str(documents[i]["id"]),
                "score": float(scores[i]),
                "text": documents[i]["text"],}
            for i in top_indices
        ]
    return retrieve
