import sys
from Datasets import resolve_datasets
from Experiment import create_experiment_dir, load_experiment
from Rag.pipeline import run_rag

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
    report = BiasReport(output_dir, [dataset_path.stem for dataset_path, *_ in datasets])
    runs = run_rag(rag_names, datasets, output_dir)
    report.finish()


if __name__ == "__main__":
    main()
