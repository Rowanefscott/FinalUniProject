# Handles connection to model being hosted in LM Studio
import json
import time
import urllib.error
import urllib.request
from Utils import merge_json

LM_STUDIO_URL = "http://localhost:1234/v1"
PAUSE_SECONDS = 60
SYSTEM_PROMPT ="Summarise the provided passages so they answer the question. Use only the passages."

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

has_generated = False

def cool_down():
    global has_generated
    if has_generated:
        print(f"Pausing {PAUSE_SECONDS}s to prevent laptop exploding")
        time.sleep(PAUSE_SECONDS)
    has_generated = True

def generate_summaries(runs, output_dir):
    model = get_active_model()
    all_summaries = {}
    for label, results in runs.items():
        summaries = {}
        for query_id, entry in results.items():
            query_text = entry["query"]
            cool_down()
            summaries[query_id] = {
                "query": query_text,
                "summary": summarise(model, query_text, entry["results"]),
            }
            print(f"Completed query {query_id}: {query_text}")
        all_summaries[label] = summaries
        output_path = output_dir / f"{label}_summaries.json"
        merge_json(summaries, output_path)
        print(f"Summaries saved to {output_path}")
    return all_summaries
