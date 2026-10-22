# Parses the TREC 2022 corpus into the standard passage format, and a seperate file for the answers/queries
# Parses in sections and appends to the same files, so a later run only parses documents not already done
import json
from itertools import islice
from pathlib import Path
from Utils import load_json, save_json

SECTION_SIZE = 1000
END_OF_FILE = b"\n]\n"

def count_parsed(output_path):
    if not output_path.exists():
        return 0
    with open(output_path, "r", encoding="utf-8") as source:
        return max(sum(1 for _ in source) - 2, 0)

def append_passages(output_path, passages):
    text = ",\n".join(json.dumps(passage, ensure_ascii=False) for passage in passages)
    if not output_path.exists() or output_path.stat().st_size == 0:
        output_path.write_text(f"[\n{text}\n]\n", encoding="utf-8")
        return
    with open(output_path, "r+b") as target:
        target.seek(-len(END_OF_FILE), 2)
        if target.read() != END_OF_FILE:
            raise ValueError(f"{output_path} does not end as expected, delete it and its answers file to reparse")
        target.seek(-len(END_OF_FILE), 2)
        target.write(f",\n{text}\n]\n".encode("utf-8"))

def to_passage(entry):
    return {
        "id": str(entry["id"]),
        "text": entry.get("plain", ""),
        "metadata": [{key: value} for key, value in entry.items() if key not in ("id", "plain")],
    }

def parse_trec2022(input_path, output_path, limit=None, answers_path=None):
    output_path = Path(output_path)
    answers_path = Path(answers_path) if answers_path else output_path.parent / f"{output_path.stem}_answers.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = count_parsed(output_path)
    if limit is not None and parsed >= limit:
        return parsed

    answers = load_json(answers_path) if answers_path.exists() else {}
    section = []

    def flush():
        nonlocal parsed
        if not section:
            return
        for passage in section:
            answers[passage["id"]] = passage["metadata"][0]["title"]
        save_json(answers, answers_path)
        append_passages(output_path, section)
        parsed += len(section)
        print(f"Parsed {len(section)} new documents ({parsed} in total)")
        section.clear()

    with open(input_path, "r", encoding="utf-8") as source:
        lines = (line for line in source if line.strip())
        for line in islice(lines, parsed, limit):
            section.append(to_passage(json.loads(line)))
            if len(section) == SECTION_SIZE:
                flush()
        flush()

    return parsed
