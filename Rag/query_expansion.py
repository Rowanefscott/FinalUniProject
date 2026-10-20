# Expands the query with terms from the top BM25 results, then searches again
import math
import re
from collections import Counter
from Rag import bm25

FEEDBACK_DOCUMENTS = 3
EXPANSION_TERMS = 10

def tokenise(text):
    return re.findall(r"[a-z]{3,}", text.lower())

def build_retriever(documents, dataset_name):
    first_pass = bm25.build_retriever(documents, dataset_name)
    document_frequency = Counter(term for document in documents for term in set(tokenise(document["text"])))

    def expand(query):
        feedback = first_pass(query, FEEDBACK_DOCUMENTS)
        term_counts = Counter(term for passage in feedback for term in tokenise(passage["text"]))
        query_terms = set(tokenise(query))
        weights = {
            term: count * math.log(len(documents) / document_frequency[term])
            for term, count in term_counts.items()
            if term not in query_terms
        }
        best = sorted(weights, key=weights.get, reverse=True)[:EXPANSION_TERMS]
        return f"{query} {' '.join(best)}"

    def retrieve(query, top_k):
        return first_pass(expand(query), top_k)

    return retrieve
