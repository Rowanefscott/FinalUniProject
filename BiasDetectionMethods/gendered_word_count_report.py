import importlib
from Config import PROJECT_DIR
from Experiment import append_run_info
from Rag.pipeline import load_documents
from Utils import merge_json

STAGE_ORDER = ["corpus", "rag", "generation"]
TRANSITIONS = [("corpus", "rag"), ("rag", "generation"), ("corpus", "generation")]

def corpus_texts(datasets):
    return {
        entry.path.stem: [document["text"] for document in load_documents(entry.path)]
        for entry in datasets
    }

def rag_texts(runs):
    return {
        label: [passage["text"] for entry in results.values() for passage in entry["results"]]
        for label, results in runs.items()
    }

def generation_texts(summaries):
    return {
        label: [entry["summary"] for entry in results.values()]
        for label, results in summaries.items()
    }

def format_value(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)

def find_dataset(label, dataset_names):
    matches = [name for name in dataset_names if label == name or label.startswith(f"{name}_")]
    return max(matches, key=len) if matches else None

class BiasReport:
    def __init__(self, output_dir, dataset_names, title):
        self.title = title
        self.output_dir = output_dir
        self.dataset_names = dataset_names
        self.stats = {}

    def run_stage(self, stage, method_names, texts_by_label):
        for method_name in method_names:
            method = importlib.import_module(f"BiasDetectionMethods.{method_name}")
            stage_stats = {label: method.measure(texts) for label, texts in texts_by_label.items()}
            self.stats.setdefault(method_name, {})[stage] = stage_stats
            lines = [f"\nBias detection: {method_name} ({stage})"]
            for label, stats in stage_stats.items():
                lines.append(f"  {label}")
                lines += [f"    {key}: {format_value(value)}" for key, value in stats.items()]
            append_run_info(self.output_dir, lines)

    def finish(self):
        for method_name, stages in self.stats.items():
            method = importlib.import_module(f"BiasDetectionMethods.{method_name}")
            lines = [f"\nBias change between stages: {method_name}"]
            for before, after in TRANSITIONS:
                if before not in stages or after not in stages:
                    continue
                for label, after_stats in stages[after].items():
                    before_label = find_dataset(label, self.dataset_names) if before == "corpus" else label
                    before_stats = stages[before].get(before_label)
                    if before_stats is None:
                        continue
                    lines.append(f"  {label}: {before} -> {after}")
                    for key in method.COMPARE:
                        start, end = before_stats[key], after_stats[key]
                        ratio = f"x{end / start:.2f}" if start else "n/a"
                        lines.append(f"    {key}: {start:.3f} -> {end:.3f} (change {end - start:+.3f}, {ratio})")
            append_run_info(self.output_dir, lines)
        merge_json({self.title: self.stats}, self.output_dir / "bias_stats.json")
