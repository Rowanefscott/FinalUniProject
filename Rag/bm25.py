# BM25 RAG method, indexes are built in sections and merged into the one index so they can be extended later
import os
import re
import shutil
import pyterrier as pt
from Config import INDEXES_DIR
from Utils import sections

SECTION_SIZE = 5000

def has_index(path):
    return (path / "data.properties").exists()

def count_indexed(index):
    return index.getCollectionStatistics().getNumberOfDocuments()

def sibling(path, suffix):
    return path.with_name(path.name + suffix)

def remove(path):
    if path.exists():
        shutil.rmtree(path)

def merge_indexes(first, second, destination):
    index_class = pt.java.autoclass("org.terrier.structures.IndexOnDisk")
    merger_class = pt.java.autoclass("org.terrier.structures.merging.StructureMerger")
    destination.mkdir()
    indexes = [
        index_class.createIndex(str(first), "data"),
        index_class.createIndex(str(second), "data"),
        index_class.createNewIndex(str(destination), "data"),
    ]
    merger_class(*indexes).mergeStructures()
    for index in indexes:
        index.close()

def add_section(index_path, documents):
    section_path, merged_path, old_path = (sibling(index_path, suffix) for suffix in ("_section", "_merged", "_old"))
    for path in (section_path, merged_path, old_path):
        remove(path)
    pt.IterDictIndexer(str(section_path), meta={"docno": 100, "text": 4096}, overwrite=True).index(
        {"docno": str(document["id"]), "text": document["text"]} for document in documents
    )
    if not has_index(index_path):
        remove(index_path)
        os.replace(section_path, index_path)
        return
    merge_indexes(index_path, section_path, merged_path)
    os.replace(index_path, old_path)
    os.replace(merged_path, index_path)
    remove(old_path)
    remove(section_path)

def load_or_extend_index(documents, dataset_name):
    index_path = (INDEXES_DIR / f"{dataset_name}_bm25_index").resolve()
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    old_path = sibling(index_path, "_old")
    if not has_index(index_path) and has_index(old_path):
        remove(index_path)
        os.replace(old_path, index_path)
    indexed = count_indexed(pt.IndexFactory.of(str(index_path))) if has_index(index_path) else 0
    for section in sections(documents[indexed:], SECTION_SIZE):
        add_section(index_path, section)
        indexed += len(section)
        print(f"BM25 index: added {len(section)} new documents, now covers {indexed} of 6475537")
    return pt.IndexFactory.of(str(index_path))

def build_retriever(documents, dataset_name):
    index = load_or_extend_index(documents, dataset_name)
    texts = {str(document["id"]): document["text"] for document in documents}
    covered = count_indexed(index)
    def retrieve(query, top_k):
        clean_query = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_query:
            return []
        fetch = top_k if covered <= len(documents) else covered
        rows = pt.BatchRetrieve(index, wmodel="BM25", num_results=fetch).search(clean_query)
        return [
            {"id": str(row["docno"]), "score": float(row["score"]), "text": texts[str(row["docno"])]}
            for _, row in rows.iterrows()
            if str(row["docno"]) in texts
        ][:top_k]
    return retrieve
