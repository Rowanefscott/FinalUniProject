# Manages datsets and their paths to be used throughout program
import re
from collections import namedtuple
from pathlib import Path
from config import DATASETS_DIR, UNPARSED_DIR
from parsers.hotpotQA import parse_hotpot
from parsers.mrsMarco import parse_mrsmarco
from parsers.trec2022 import parse_trec2022

PARSERS = {
    "msmarco": ("msmarco.jsonl", parse_mrsmarco, False),
    "hotpotQA": ("hotpotQA.json", parse_hotpot, False),
    "trec2022": ("trecCorpus20220301Plain.json", parse_trec2022, True),
}

def find_parser_name(name):
    return next((registered for registered in PARSERS if registered.lower() == name.lower()), name)

def selection_limit(selection):
    numbers = re.findall(r"\d+", str(selection))
    return max(map(int, numbers)) if numbers else None

def get_dataset(name, limit=None):
    name = find_parser_name(name)
    output_path = DATASETS_DIR / f"{name}.json"
    partial = name in PARSERS and PARSERS[name][2]
    if output_path.exists() and not (partial and limit):
        return output_path
    if name not in PARSERS:
        raise ValueError(f"No parser registered for '{name}'")
    input_file, parser, _ = PARSERS[name]
    input_path = UNPARSED_DIR / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    print("Parsing data file")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if partial:
        parser(input_path, output_path, limit)
    else:
        parser(input_path, output_path)
    return output_path

DataEntry = namedtuple("DataEntry", "path queries selection limit")

def resolve_datasets(entries):
    resolved = []
    for dataset, queries, *selection in entries:
        selection = selection[0] if selection else ""
        limit = selection_limit(selection)
        partial = PARSERS.get(find_parser_name(dataset), (None, None, False))[2]
        resolved.append(DataEntry(
            get_dataset(dataset, limit),
            DATASETS_DIR / Path(queries).with_suffix(".json"),
            selection,
            limit if partial else None,
        ))
    return resolved

def select_queries(queries, selection=None):
    if not selection:
        return queries
    items = list(queries.items())
    chosen = []
    for part in str(selection).split(","):
        start, _, end = part.strip().partition("-")
        chosen += items[int(start) - 1:int(end or start)]
    return dict(chosen)
