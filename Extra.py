import json
from pathlib import Path
from Parsers.mrsmarco import parse_mrsmarco

""" =================== HANDLES PARSING DATESET TO STANDARD FORMAT =================== """

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("Data")
PARSERS = {}

# Defines which parser to use
def register_parser(name, input_file, parser):
    PARSERS[name] = (input_file, parser)

register_parser("msmarco", "msmarco.jsonl", parse_mrsmarco)

# Checks if dataset already exists before parsing
def get_dataset(name):
    output_path = DATA_DIR / f"{name}.txt"

    if output_path.exists():
        return output_path

    if name not in PARSERS:
        raise ValueError(f"No parser registered for '{name}'")

    print ("Parsing Data File")
    input_file, parser = PARSERS[name]
    input_path = PROJECT_DIR / "Unparsed" / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parser(input_path, output_path)

    return output_path

""" =================== LOADS EXPERIMENT DETAILS =================== """

# Loads experiment details from experiment.txt
def load_experiment(name):
    with open("experiments.json", "r", encoding="utf-8") as file:
        experiments = json.load(file)

    if name not in experiments:
        raise ValueError(f"Experiment '{name}' not found")

    return experiments[name]