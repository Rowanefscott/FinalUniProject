import sys
from Extra import load_experiment, get_dataset

def main():
    if len(sys.argv) != 2:
        raise SystemExit("python main.py <experiment_name>")

    config = load_experiment(sys.argv[1])

    datasets = [
        get_dataset(name)
        for name in config.get("Data", [])
    ]

    print(config)
    print(datasets)


if __name__ == "__main__":
    main()


