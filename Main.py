import sys
from BiasDetectionMethods.gendered_word_count_report import BiasReport, corpus_texts, generation_texts, rag_texts
from Datasets import resolve_datasets
from Experiment import append_run_info, create_experiment_dir, get_methods, is_enabled, load_experiment
from Generation import generate_summaries
from Rag.pipeline import run_rag, use_all_documents

def run_entry(entry, config, rag_names, output_dir):
    title = f"{entry.path.stem} | {entry.queries.stem} | queries: {entry.selection or 'all'}"
    append_run_info(output_dir, ["=====================================", f"Data: {title}"])
    report = BiasReport(output_dir, [entry.path.stem], title)

    methods = get_methods(config, "corpusBiasTesting")
    if methods:
        report.run_stage("corpus", methods, corpus_texts([entry]))

    runs = run_rag(rag_names, [entry], output_dir)
    methods = get_methods(config, "ragBiasTesting")
    if methods:
        report.run_stage("rag", methods, rag_texts(runs))

    if is_enabled(config, "LLMGeneration"):
        summaries = generate_summaries(runs or use_all_documents([entry]), output_dir)
        methods = get_methods(config, "generationBiasTesting")
        if methods:
            report.run_stage("generation", methods, generation_texts(summaries))

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
