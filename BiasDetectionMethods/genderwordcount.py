# Compares % of gendered nouns in english language with those found in input/retrieved/generated text
import re
from bisect import bisect_left
from collections import Counter
from functools import lru_cache
from config import GENDERED_NOUNS_FILE

COMPARE = ["gendered_percent", "male_percent", "female_percent", "skew"]
TOP_TERMS = 5

@lru_cache(maxsize=1)
def load_gendered_nouns():
    with open(GENDERED_NOUNS_FILE, "r", encoding="utf-8") as source:
        pairs = sorted(
            (term.replace("_", " "), gender)
            for term, gender in (line.strip().lower().split(",") for line in source if "," in line)
        )
    terms = [term for term, _ in pairs]
    max_length = max(len(term.split()) for term in terms)
    return terms, [gender for _, gender in pairs], max_length

def lookup_gender(term, terms, genders):
    position = bisect_left(terms, term)
    if position < len(terms) and terms[position] == term:
        return genders[position]
    return None

def find_gendered_terms(tokens):
    terms, genders, max_length = load_gendered_nouns()
    found = []
    position = 0
    while position < len(tokens):
        for length in range(max_length, 0, -1):
            term = " ".join(tokens[position:position + length])
            gender = lookup_gender(term, terms, genders)
            if gender in ("m", "f"):
                found.append((term, gender))
                position += length
                break
        else:
            position += 1
    return found

def percent(part, whole):
    return 100 * part / whole if whole else 0.0

def format_top(counter):
    return ", ".join(f"{term}:{count}" for term, count in counter.most_common(TOP_TERMS)) or "none"

def measure(texts):
    total_words = 0
    terms_by_gender = {"m": Counter(), "f": Counter()}
    documents_with_terms = 0
    document_percents = []
    for text in texts:
        tokens = re.findall(r"[a-z']+", text.lower())
        found = find_gendered_terms(tokens)
        total_words += len(tokens)
        documents_with_terms += bool(found)
        document_percents.append(percent(len(found), len(tokens)))
        for term, gender in found:
            terms_by_gender[gender][term] += 1
    male = sum(terms_by_gender["m"].values())
    female = sum(terms_by_gender["f"].values())
    return {
        "documents": len(document_percents),
        "total_words": total_words,
        "gendered_words": male + female,
        "male_words": male,
        "female_words": female,
        "gendered_percent": percent(male + female, total_words),
        "male_percent": percent(male, total_words),
        "female_percent": percent(female, total_words),
        "male_to_female_ratio": male / female if female else None,
        "skew": (male - female) / (male + female) if male + female else 0.0,
        "unique_gendered_terms": len(terms_by_gender["m"]) + len(terms_by_gender["f"]),
        "documents_with_gendered_terms_percent": percent(documents_with_terms, len(document_percents)),
        "mean_document_gendered_percent": sum(document_percents) / len(document_percents) if document_percents else 0.0,
        "top_male_terms": format_top(terms_by_gender["m"]),
        "top_female_terms": format_top(terms_by_gender["f"]),
    }
