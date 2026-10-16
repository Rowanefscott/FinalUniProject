# Loads the experiment specification, notes time/date and organises into folder
from datetime import datetime
from Config import EXPERIMENTS_DIR, EXPERIMENTS_FILE
from Utils import load_json

def load_experiment(name):
    experiments = load_json(EXPERIMENTS_FILE)

    if name not in experiments:
        raise ValueError(f"Experiment '{name}' not found")
    return experiments[name]

def create_experiment_dir(name, description=""):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    folder = EXPERIMENTS_DIR / name
    count = 2
    while folder.exists():
        folder = EXPERIMENTS_DIR / f"{name}{count}"
        count += 1

    folder.mkdir()
    (folder / "run_info.txt").write_text(
        f"Experiment: {name}\n"
        f"Run at: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Description: {description}\n",
        encoding="utf-8",
    )
    return folder

def append_run_info(folder, lines):
    with open(folder / "run_info.txt", "a", encoding="utf-8") as target:
        target.write("\n".join(lines) + "\n")

def is_enabled(config, key):
    return any(
        name.lower() == key.lower() and str(value).lower() == "true"
        for name, value in config.items()
    )
