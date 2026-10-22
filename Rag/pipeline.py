# Manages chosen rag method/s
import importlib
from Datasets import select_queries
from Rag.topk import resolve_top_k, top_k_label
from Utils import load_json, merge_json

def load_documents(dataset_path):
    documents = [
        document for document in load_json(dataset_path)
        if document.get("text", "").strip()
    ]
    if not documents:
        raise ValueError(f"No documents found in {dataset_path}")
    return documents

def load_entry_documents(entry):
    return load_documents(entry.path)[:entry.limit]

def search_queries(retrieve, queries, top_k):
    results = {}
    for query_id, query_text in queries.items():
        results[query_id] = {
            "query": query_text,
            "results": retrieve(query_text, top_k),
        }
    return results

def use_all_documents(datasets):
    runs = {}
    for entry in datasets:
        documents = load_entry_documents(entry)
        queries = select_queries(load_json(entry.queries), entry.selection)
        runs[entry.path.stem] = {
            query_id: {"query": query_text, "results": documents}
            for query_id, query_text in queries.items()
        }
    return runs

retrievers = {}

def get_retriever(rag_name, rag, entry):
    key = (rag_name, entry.path.stem, entry.limit)
    if key not in retrievers:
        retrievers[key] = rag.build_retriever(load_entry_documents(entry), entry.path.stem)
    return retrievers[key]

def run_rag(rag_names, datasets, output_dir, top_ks=(None,)):
    runs = {}
    for rag_name in rag_names:
        rag = importlib.import_module(f"Rag.{rag_name}")
        for entry in datasets:
            retrieve = get_retriever(rag_name, rag, entry)
            queries = select_queries(load_json(entry.queries), entry.selection)
            for top_k in top_ks:
                label = f"{entry.path.stem}_{rag_name}{top_k_label(top_k)}"
                results = search_queries(retrieve, queries, resolve_top_k(top_k))
                merge_json(results, output_dir / f"{label}_results.json")
                runs[label] = results
    return runs
