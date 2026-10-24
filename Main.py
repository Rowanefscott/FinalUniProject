import sys
from biasReport import BiasReport, corpus_texts, generated_texts, retrieved_texts
from datasetManager import resolve_datasets
from experiment import append_run_info, create_experiment_dir, get_stage_list, is_enabled, load_experiment
from generation import generate_summaries
from rag.topK import top_k_values
from rag.pipeline import run_rag, use_all_documents

NO_DETECTION = ("normal", "none", "false")

def run_bias_stage(report, config, stage, get_texts, cache_key=None):
    methods = get_stage_list(config, "biasDetectionMethod", stage, NO_DETECTION)
    if methods:
        tests = get_stage_list(config, "biasTest", stage, ("none", "false")) or ["normal"]
        report.run_stage(stage, tests, methods, get_texts, cache_key)

def run_entry(entry, config, rag_names, output_dir, shared_cache):
    title = f"{entry.path.stem} | {entry.queries.stem} | queries: {entry.selection or 'all'}"
    append_run_info(output_dir, ["=====================================", f"Data: {title}"])
    report = BiasReport(output_dir, [entry.path.stem], title, shared_cache)

    run_bias_stage(report, config, "corpus", lambda: corpus_texts([entry]), (entry.path.stem, entry.limit))

    runs = run_rag(rag_names, [entry], output_dir, top_k_values(config))
    run_bias_stage(report, config, "retrieved", lambda: retrieved_texts(runs))

    if is_enabled(config, "LLMGeneration"):
        summaries = generate_summaries(runs or use_all_documents([entry]), output_dir)
        run_bias_stage(report, config, "generated", lambda: generated_texts(summaries))

    report.finish()

def main():
    if len(sys.argv) != 2:
        raise SystemExit("python Main.py <experiment_name>")
    name = sys.argv[1]
    config = load_experiment(name)
    rag_names = config.get("Rag", [])

    if isinstance(rag_names, str):
        rag_names = [rag_names]
    datasets = resolve_datasets(config.get("Data", []))
    output_dir = create_experiment_dir(name, config.get("Description", ""))
    shared_cache = {}
    for entry in datasets:
        run_entry(entry, config, rag_names, output_dir, shared_cache)

if __name__ == "__main__":
    main()
