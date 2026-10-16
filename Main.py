import sys
from Experiment import  load_experiment
from Rag.pipeline import run_rag

def main():
    if len(sys.argv) != 2:
        raise SystemExit("python Main.py <experiment_name>")

    name = sys.argv[1]
    config = load_experiment(name)

    rag_names = config.get("Rag", [])
    if isinstance(rag_names, str):
        rag_names = [rag_names]
    run_rag(rag_names, datasets, output_dir)

if __name__ == "__main__":
    main()
