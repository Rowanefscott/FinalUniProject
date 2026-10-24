# Parses HotpotQA entries into individual passages and separate answers
import json
from pathlib import Path
import string

def parse_hotpot(input_path, output_path, answers_path=None):
    output_path = Path(output_path)
    if answers_path is None:
        answers_path = output_path.parent / "hotpotQAAnswers.json"
    else:
        answers_path = Path(answers_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as source:
        content = source.read()
    try:
        entries = json.loads(content)
    except json.JSONDecodeError:
        entries = [json.loads(line) for line in content.splitlines() if line.strip()]
    answers = {}
    passages = []

    for question_number, entry in enumerate(entries, start=1):
        question_id = str(question_number)
        answers[question_id] = entry.get("answer", "")
        context = entry.get("context", {})
        titles = context.get("title", [])
        paragraphs = context.get("sentences", [])
        supporting_titles = set(entry.get("supporting_facts", {}).get("title", []))

        for paragraph_index, sentences in enumerate(paragraphs):
            title = (titles[paragraph_index] if paragraph_index < len(titles) else "")
            letter = (string.ascii_lowercase[paragraph_index] if paragraph_index < 26 else str(paragraph_index + 1))
            passages.append({
                "id": f"{question_id}.{letter}",
                "text": " ".join(sentences).strip(),
                "metadata": [{"question_id": entry.get("id")}, {"question_number": question_number}, {"question": entry.get("question", "")}, {"title": title}, {"paragraph_index": paragraph_index}, {"is_supporting": title in supporting_titles},],
            })

    with open(output_path, "w", encoding="utf-8") as target:
        json.dump(passages, target, ensure_ascii=False, indent=2)

    with open(answers_path, "w", encoding="utf-8") as target:
        json.dump(answers, target, ensure_ascii=False, indent=2)

    return len(passages)