import sys
from Datasets import resolve_datasets
from Experiment import create_experiment_dir, is_enabled, load_experiment
from Generation import generate_summaries
from Rag.pipeline import run_rag, use_all_documents

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

    runs = run_rag(rag_names, datasets, output_dir)

    if is_enabled(config, "LLMGeneration"):
        summaries = generate_summaries(runs or use_all_documents(datasets), output_dir)

if __name__ == "__main__":
    main()
