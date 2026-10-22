# General JSON functions
import importlib
import json
import pkgutil

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

def merge_json(data, path):
    existing = load_json(path) if path.exists() else {}
    save_json({**existing, **data}, path)


def normalise_name(name):
    return name.lower().replace("_", "").replace(" ", "")


def load_component(package, name):
    available = [module.name for module in pkgutil.iter_modules(importlib.import_module(package).__path__)]
    match = next((module for module in available if normalise_name(module) == normalise_name(name)), None)
    if match is None:
        raise ValueError(f"No '{name}' in {package}. Available: {', '.join(available)}")
    return importlib.import_module(f"{package}.{match}")


def sections(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
