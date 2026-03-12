import json
from pathlib import Path

from src.results import display_all_confusion_matrices
from src.results import print_model_results

def generate_plots():
    results_file = Path("outputs/results.json")

    with open(results_file, 'r') as file:
        results_data = json.load(file)

    display_all_confusion_matrices(results_data)
    print_model_results(results_data)



if __name__ == "__main__":
    generate_plots()