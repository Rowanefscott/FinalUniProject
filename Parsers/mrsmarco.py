# Parses msmarco corpus entries into a standard JSON format
import json
from pathlib import Path

def parse_mrsmarco(input_path, output_path=None):
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "msmarco.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    passages = []

    with open(input_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            parsed = {
                "id": entry.get("_id", entry.get("id")),
                "text": entry.get("text", ""),
                "metadata": [{key: value} for key, value in entry.items() if key not in ("_id", "id", "text")],}
            passages.append(parsed)

    with open(output_path, "w", encoding="utf-8") as target:
        json.dump(passages, target, ensure_ascii=False, indent=2)

    return len(passages)