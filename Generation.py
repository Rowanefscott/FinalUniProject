# Handles connection to model being hosted in LM Studio
import json
import urllib.error
import urllib.request
from Utils import save_json

LM_STUDIO_URL = "http://localhost:1234/v1"
SYSTEM_PROMPT = "Summarise the provided passages so they answer the question. Use only the passages."

def request(path, payload=None):
    http_request = urllib.request.Request(
        LM_STUDIO_URL + path,
        data=json.dumps(payload).encode("utf-8") if payload else None,
        headers={"Content-Type": "application/json"},)
    with urllib.request.urlopen(http_request, timeout=600) as response:
        return json.load(response)

def get_active_model():
    try:
        return request("/models")["data"][0]["id"]
    except urllib.error.URLError:
        raise SystemExit(f"Could not reach LM Studio at {LM_STUDIO_URL}. Start its local server and load a model.")
    except IndexError:
        raise SystemExit("LM Studio has no model loaded.")

def summarise(model, query, passages):
    context = "\n\n".join(f"[{passage['id']}] {passage['text']}" for passage in passages)
    reply = request("/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nPassages:\n{context}"},],})
    return reply["choices"][0]["message"]["content"]

def generate_summaries(runs, output_dir):
    model = get_active_model()
    all_summaries = {}
    for label, results in runs.items():
        summaries = {}
        for query_id, entry in results.items():
            summaries[query_id] = {
                "query": entry["query"],
                "summary": summarise(model, entry["query"], entry["results"]),
            }
            print(f"Summarised query {query_id} ({label})")
        all_summaries[label] = summaries
        output_path = output_dir / f"{label}_summaries.json"
        save_json(summaries, output_path)
        print(f"Summaries saved to {output_path}")
    return all_summaries
