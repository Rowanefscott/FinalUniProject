# Manages chosen rag method/s
import importlib
from Datasets import select_queries
from Utils import load_json, merge_json

TOP_K = 10

def load_documents(dataset_path):
    documents = [
        document for document in load_json(dataset_path)
        if document.get("text", "").strip()
    ]
    if not documents:
        raise ValueError(f"No documents found in {dataset_path}")
    return documents

def search_queries(retrieve, queries):
    results = {}
    for query_id, query_text in queries.items():
        results[query_id] = {
            "query": query_text,
            "results": retrieve(query_text, TOP_K),
        }
    return results

def use_all_documents(datasets):
    runs = {}
    for entry in datasets:
        documents = load_documents(entry.path)
        queries = select_queries(load_json(entry.queries), entry.selection)
        runs[entry.path.stem] = {
            query_id: {"query": query_text, "results": documents}
            for query_id, query_text in queries.items()
        }
    return runs

retrievers = {}

def get_retriever(rag_name, rag, entry):
    key = (rag_name, entry.path.stem)
    if key not in retrievers:
        retrievers[key] = rag.build_retriever(load_documents(entry.path), entry.path.stem)
    return retrievers[key]

def run_rag(rag_names, datasets, output_dir):
    runs = {}
    for rag_name in rag_names:
        rag = importlib.import_module(f"Rag.{rag_name}")
        for entry in datasets:
            retrieve = get_retriever(rag_name, rag, entry)
            queries = select_queries(load_json(entry.queries), entry.selection)
            results = search_queries(retrieve, queries)
            output_path = output_dir / f"{entry.path.stem}_{rag_name}_results.json"
            merge_json(results, output_path)
            runs[f"{entry.path.stem}_{rag_name}"] = results
    return runs
