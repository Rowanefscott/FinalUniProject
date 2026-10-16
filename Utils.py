# For loading saving and path finding
import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as source:
        content = source.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
