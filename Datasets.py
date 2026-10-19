# Manages datsets and their paths to be used throughout program
from pathlib import Path
from Config import DATASETS_DIR, UNPARSED_DIR
from Parsers.hotpotQA import parse_hotpot
from Parsers.mrsmarco import parse_mrsmarco

PARSERS = {
    "msmarco": ("msmarco.jsonl", parse_mrsmarco),
    "hotpotQA": ("hotpotQA.json", parse_hotpot),
}

def get_dataset(name):
    output_path = DATASETS_DIR / f"{name}.json"
    if output_path.exists():
        return output_path
    if name not in PARSERS:
        raise ValueError(f"No parser registered for '{name}'")
    input_file, parser = PARSERS[name]
    input_path = UNPARSED_DIR / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    print("Parsing data file")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    parser(input_path, output_path)
    return output_path

def resolve_datasets(entries):
    return [
        (get_dataset(dataset), DATASETS_DIR / Path(queries).with_suffix(".json"), *selection)
        for dataset, queries, *selection in entries
    ]

def select_queries(queries, selection=None):
    if not selection:
        return queries
    items = list(queries.items())
    chosen = []
    for part in str(selection).split(","):
        start, _, end = part.strip().partition("-")
        chosen += items[int(start) - 1:int(end or start)]
    return dict(chosen)
