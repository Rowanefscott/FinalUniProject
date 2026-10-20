# Reorders BM25 docs with a cross encoder
from sentence_transformers import CrossEncoder
from Rag import bm25

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 50

def build_retriever(documents, dataset_name):
    first_stage = bm25.build_retriever(documents, dataset_name)
    reranker = CrossEncoder(MODEL_NAME)

    def retrieve(query, top_k):
        candidates = first_stage(query, max(CANDIDATES, top_k))
        if not candidates:
            return []
        scores = reranker.predict([[query, candidate["text"]] for candidate in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [{"id": passage["id"], "score": float(score), "text": passage["text"]} for passage, score in ranked]

    return retrieve
