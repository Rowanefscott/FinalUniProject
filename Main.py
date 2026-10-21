import sys
from BiasReport import BiasReport, corpus_texts, generated_texts, retrieved_texts
from Datasets import resolve_datasets
from Experiment import append_run_info, create_experiment_dir, get_stage_list, is_enabled, load_experiment
from Generation import generate_summaries
from Rag.topk import top_k_values
from Rag.pipeline import run_rag, use_all_documents

NO_DETECTION = ("normal", "none", "false")

def run_bias_stage(report, config, stage, get_texts):
    methods = get_stage_list(config, "biasDetectionMethod", stage, NO_DETECTION)
    if methods:
        tests = get_stage_list(config, "biasTest", stage, ("none", "false")) or ["normal"]
        report.run_stage(stage, tests, methods, get_texts())

def run_entry(entry, config, rag_names, output_dir):
    title = f"{entry.path.stem} | {entry.queries.stem} | queries: {entry.selection or 'all'}"
    append_run_info(output_dir, ["=====================================", f"Data: {title}"])
    report = BiasReport(output_dir, [entry.path.stem], title)

    run_bias_stage(report, config, "corpus", lambda: corpus_texts([entry]))

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
    for entry in datasets:
        run_entry(entry, config, rag_names, output_dir)

if __name__ == "__main__":
    main()
