import json
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np

from src.config import VOTING_FILE_NAME

def print_results(results, models):
    
    for j, model_name in enumerate(models.keys()):
        mean_scores = results[j].mean(axis=0)
        std_scores = results[j].std(axis=0)

        print(f"{model_name}")
        print(f"Accuracy:  {mean_scores[0]:.4f} (± {std_scores[0]:.4f})")
        print(f"Precision: {mean_scores[1]:.4f} (± {std_scores[1]:.4f})")
        print(f"Recall:    {mean_scores[2]:.4f} (± {std_scores[2]:.4f})")
        print(f"F1 Score:  {mean_scores[3]:.4f} (± {std_scores[3]:.4f})")

def save_result(data):

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / VOTING_FILE_NAME

    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved to {json_filepath}")

#Plotting graphs
def display_all_confusion_matrices(evaluation_results):
    model_names = list(evaluation_results.keys())
    num_models = len(model_names)
    
    cols = 3
    rows = (num_models + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for idx, model_name in enumerate(model_names):
        all_y_true = []
        all_y_pred = []
        for fold_result in evaluation_results[model_name]:
            all_y_true.extend(fold_result['y_true'])
            all_y_pred.extend(fold_result['y_pred'])

        cm = confusion_matrix(all_y_true, all_y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=axes[idx], colorbar=False)
        axes[idx].set_title(model_name, fontsize=11, fontweight='bold')

    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices — All Classifiers", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/all_confusion_matrices.png', bbox_inches='tight')
    plt.show()
    print("Saved to outputs/all_confusion_matrices.png")

