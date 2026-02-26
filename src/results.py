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

def display_confusion_matrix():
    with open("outputs/" + VOTING_FILE_NAME, 'r') as f:
        data = json.load(f)
    
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    cm = confusion_matrix(y_true, y_pred)
    
    _, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    plt.title(f"Weighted Voting Confusion Matrix")
    plt.savefig('outputs/confusion_matrix.png')

def save_voting_result(data):
    all_true = np.concatenate([fold['y_true'] for fold in data])
    all_pred = np.concatenate([fold['y_pred'] for fold in data])

    final_data = {
        "model_name": "WeightedVoting",
        "total_samples": len(all_true),
        "y_true": all_true.tolist(),
        "y_pred": all_pred.tolist()
    }

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filepath = output_dir / VOTING_FILE_NAME

    with open(json_filepath, 'w') as f:
        json.dump(final_data, f, indent=4)
    
    print(f"Saved to {json_filepath}")