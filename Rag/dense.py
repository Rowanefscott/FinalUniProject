# Dense Embeddings RAG method, embeddings are computed in sections and saved to the one file so they can be extended later
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from Config import INDEXES_DIR
from Utils import sections

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SECTION_SIZE = 500

def load_or_extend_embeddings(model, documents, dataset_name):
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEXES_DIR / f"{dataset_name}_dense_embeddings.npy"
    temporary_path = INDEXES_DIR / f"{dataset_name}_dense_embeddings_tmp.npy"
    embeddings = np.load(path) if path.exists() else np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    for section in sections(documents[len(embeddings):], SECTION_SIZE):
        new = model.encode_document([document["text"] for document in section], normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.vstack([embeddings, new])
        np.save(temporary_path, embeddings)
        os.replace(temporary_path, path)
        print(f"Embeddings: added {len(new)} new documents, now cover {len(embeddings)} of 6475537")
    return embeddings[:len(documents)]

def build_retriever(documents, dataset_name):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = load_or_extend_embeddings(model, documents, dataset_name)
    def retrieve(query, top_k):
        scores = np.dot(embeddings, model.encode_query(query, normalize_embeddings=True))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"id": str(documents[i]["id"]),
                "score": float(scores[i]),
                "text": documents[i]["text"],}
            for i in top_indices
        ]
    return retrieve
