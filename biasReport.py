# Runs bias tests (data transforms) then bias detection methods per stage, and reports the results
from experiment import append_run_info
from rag.pipeline import load_entry_documents
from utils import load_component, merge_json

TRANSITIONS = [("corpus", "retrieved"), ("retrieved", "generated"), ("corpus", "generated")]

def corpus_texts(datasets):
    return {
        entry.path.stem: [document["text"] for document in load_entry_documents(entry)]
        for entry in datasets
    }

def retrieved_texts(runs):
    return {
        label: [passage["text"] for entry in results.values() for passage in entry["results"]]
        for label, results in runs.items()
    }

def generated_texts(summaries):
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

def short_name(module):
    return module.__name__.split(".")[-1]

def comparison_lines(method, before_stats, after_stats):
    lines = []
    for key in method.COMPARE:
        start, end = before_stats[key], after_stats[key]
        ratio = f"x{end / start:.2f}" if start else "n/a"
        lines.append(f"    {key}: {start:.3f} -> {end:.3f} (change {end - start:+.3f}, {ratio})")
    return lines

class BiasReport:
    def __init__(self, output_dir, dataset_names, title, cache=None):
        self.title = title
        self.cache = {} if cache is None else cache  # shared between entries so identical stages are only measured once
        self.output_dir = output_dir
        self.dataset_names = dataset_names
        self.stats = {}  # method -> stage -> test -> label -> stats

    def run_stage(self, stage, test_names, method_names, get_texts, cache_key=None):
        methods = [load_component("biasDetectionMethods", name) for name in method_names]
        texts_by_label = None
        for test in (load_component("biasTests", name) for name in test_names):
            transformed = None
            for method in methods:
                key = (cache_key, short_name(test), short_name(method))
                if cache_key is not None and key in self.cache:
                    stage_stats = self.cache[key]
                    self.stats.setdefault(short_name(method), {}).setdefault(stage, {})[short_name(test)] = stage_stats
                    append_run_info(self.output_dir, [f"\nBias detection: {short_name(method)} ({stage}, {short_name(test)} test): same as reported earlier for {', '.join(stage_stats)}"])
                    continue
                if transformed is None:
                    texts_by_label = texts_by_label or get_texts()
                    transformed = {label: test.transform(texts) for label, texts in texts_by_label.items()}
                stage_stats = {label: method.measure(texts) for label, texts in transformed.items()}
                self.stats.setdefault(short_name(method), {}).setdefault(stage, {})[short_name(test)] = stage_stats
                if cache_key is not None:
                    self.cache[key] = stage_stats
                lines = [f"\nBias detection: {short_name(method)} ({stage}, {short_name(test)} test)"]
                for label, stats in stage_stats.items():
                    lines.append(f"  {label}")
                    lines += [f"    {key}: {format_value(value)}" for key, value in stats.items()]
                append_run_info(self.output_dir, lines)

    def finish(self):
        for method_name, stages in self.stats.items():
            method = load_component("biasDetectionMethods", method_name)
            self.report_stage_changes(method, method_name, stages)
            self.report_test_differences(method, method_name, stages)
        merge_json({self.title: self.stats}, self.output_dir / "biasStats.json")

    def report_stage_changes(self, method, method_name, stages):
        for before, after in TRANSITIONS:
            for test_name in stages.get(after, {}):
                if test_name not in stages.get(before, {}):
                    continue
                lines = [f"\nBias change between stages: {method_name} ({test_name} test)"]
                for label, after_stats in stages[after][test_name].items():
                    before_label = find_dataset(label, self.dataset_names) if before == "corpus" else label
                    before_stats = stages[before][test_name].get(before_label)
                    if before_stats is not None:
                        lines.append(f"  {label}: {before} -> {after}")
                        lines += comparison_lines(method, before_stats, after_stats)
                append_run_info(self.output_dir, lines)

    def report_test_differences(self, method, method_name, stages):
        for stage, tests in stages.items():
            if "normal" not in tests:
                continue
            for test_name, test_stats in tests.items():
                if test_name == "normal":
                    continue
                lines = [f"\nBias difference between tests: {method_name} ({stage}, normal -> {test_name})"]
                for label, stats in test_stats.items():
                    lines.append(f"  {label}")
                    lines += comparison_lines(method, tests["normal"][label], stats)
                append_run_info(self.output_dir, lines)
