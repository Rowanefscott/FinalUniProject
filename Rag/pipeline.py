# Manages chosen rag method/s
import importlib
from Datasets import select_queries
from Utils import load_json, save_json

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
        print(f"Completed query {query_id}: {query_text}")
    return results

def use_all_documents(datasets):
    runs = {}
    for dataset_path, query_path, *selection in datasets:
        documents = load_documents(dataset_path)
        queries = select_queries(load_json(query_path), *selection)
        runs[dataset_path.stem] = {
            query_id: {"query": query_text, "results": documents}
            for query_id, query_text in queries.items()
        }
    return runs

def run_rag(rag_names, datasets, output_dir):
    runs = {}
    for rag_name in rag_names:
        rag = importlib.import_module(f"Rag.{rag_name}")
        for dataset_path, query_path, *selection in datasets:
            documents = load_documents(dataset_path)
            retrieve = rag.build_retriever(documents, dataset_path.stem)
            queries = select_queries(load_json(query_path), *selection)
            results = search_queries(retrieve, queries)
            output_path = output_dir / f"{dataset_path.stem}_{rag_name}_results.json"
            save_json(results, output_path)
            print(f"Results saved to {output_path}")
            runs[f"{dataset_path.stem}_{rag_name}"] = results
    return runs
