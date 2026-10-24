# Uses BM25 with dense embeddings
from rag import bm25, dense

CANDIDATES = 50
RRF_K = 60

def build_retriever(documents, dataset_name):
    retrievers = [
        bm25.build_retriever(documents, dataset_name),
        dense.build_retriever(documents, dataset_name),
    ]

    def retrieve(query, top_k):
        scores = {}
        passages = {}
        for retriever in retrievers:
            for rank, passage in enumerate(retriever(query, max(CANDIDATES, top_k)), start=1):
                scores[passage["id"]] = scores.get(passage["id"], 0.0) + 1 / (RRF_K + rank)
                passages[passage["id"]] = passage
        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [{"id": id_, "score": scores[id_], "text": passages[id_]["text"]} for id_ in ranked]

    return retrieve
