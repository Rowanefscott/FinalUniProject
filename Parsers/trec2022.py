# Parses the TREC 2022 corpus into the standard passage format, and a seperate file for the answers/queries
import json
from pathlib import Path

def parse_trec2022(input_path, output_path, answers_path=None):
    output_path = Path(output_path)
    answers_path = Path(answers_path) if answers_path else output_path.parent / f"{output_path.stem}_answers.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    passages = []
    answers = {}

    with open(input_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

            passage_id = str(entry["id"])
            answers[passage_id] = entry.get("title", "")
            passages.append({
                "id": passage_id,
                "text": entry.get("plain", ""),
                "metadata": [
                    {key: value}
                    for key, value in entry.items()
                    if key not in ("id", "plain")
                ],
            })

    with open(output_path, "w", encoding="utf-8") as target:
        json.dump(passages, target, ensure_ascii=False, indent=2)

    with open(answers_path, "w", encoding="utf-8") as target:
        json.dump(answers, target, ensure_ascii=False, indent=2)

    return len(passages)
