# BM25 RAG method
import re
import shutil
import pyterrier as pt
from Config import INDEXES_DIR

def load_or_build_index(documents, dataset_name):
    index_path = (INDEXES_DIR / f"{dataset_name}_bm25_index").resolve()
    if (index_path / "data.properties").exists():
        return pt.IndexFactory.of(str(index_path))
    if index_path.exists():
        shutil.rmtree(index_path)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    indexer = pt.IterDictIndexer(
        str(index_path),
        meta={"docno": 100, "text": 4096},
        overwrite=True,
    )
    index_ref = indexer.index(
        {"docno": str(document["id"]), "text": document["text"]}
        for document in documents
    )
    return pt.IndexFactory.of(index_ref)

def build_retriever(documents, dataset_name):
    index = load_or_build_index(documents, dataset_name)
    texts = {str(document["id"]): document["text"] for document in documents}
    def retrieve(query, top_k):
        clean_query = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_query:
            return []
        rows = pt.BatchRetrieve(index, wmodel="BM25", num_results=top_k).search(clean_query)
        return [
            {
                "id": str(row["docno"]),
                "score": float(row["score"]),
                "text": texts[str(row["docno"])],
            }
            for _, row in rows.iterrows()
        ]
    return retrieve
