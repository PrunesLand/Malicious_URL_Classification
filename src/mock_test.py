import numpy as np
from src.results import display_all_confusion_matrices, save_voting_result

def test_mock_voting():
    mock_evaluation_results = {
        "RandomForest": [
            {"fold": 1, "y_true": np.array([0, 1, 0, 1, 1]), "y_pred": np.array([0, 1, 1, 1, 0]), "metrics": {"accuracy": 0.8, "precision": 0.83, "recall": 0.8, "f1": 0.80}},
            {"fold": 2, "y_true": np.array([1, 0, 1, 1, 0]), "y_pred": np.array([1, 0, 1, 0, 0]), "metrics": {"accuracy": 0.8, "precision": 0.80, "recall": 0.8, "f1": 0.79}},
            {"fold": 3, "y_true": np.array([0, 0, 0, 1, 1]), "y_pred": np.array([0, 0, 0, 1, 1]), "metrics": {"accuracy": 1.0, "precision": 1.00, "recall": 1.0, "f1": 1.00}},
        ],
        "LogisticRegression": [
            {"fold": 1, "y_true": np.array([0, 1, 0, 1, 1]), "y_pred": np.array([0, 0, 0, 1, 1]), "metrics": {"accuracy": 0.8, "precision": 0.82, "recall": 0.8, "f1": 0.80}},
            {"fold": 2, "y_true": np.array([1, 0, 1, 1, 0]), "y_pred": np.array([1, 0, 0, 1, 0]), "metrics": {"accuracy": 0.8, "precision": 0.78, "recall": 0.8, "f1": 0.78}},
            {"fold": 3, "y_true": np.array([0, 0, 0, 1, 1]), "y_pred": np.array([0, 0, 1, 1, 1]), "metrics": {"accuracy": 0.8, "precision": 0.80, "recall": 0.8, "f1": 0.79}},
        ],
        "SVM": [
            {"fold": 1, "y_true": np.array([0, 1, 0, 1, 1]), "y_pred": np.array([0, 1, 0, 1, 1]), "metrics": {"accuracy": 1.0, "precision": 1.00, "recall": 1.0, "f1": 1.00}},
            {"fold": 2, "y_true": np.array([1, 0, 1, 1, 0]), "y_pred": np.array([1, 1, 1, 0, 0]), "metrics": {"accuracy": 0.6, "precision": 0.60, "recall": 0.6, "f1": 0.58}},
            {"fold": 3, "y_true": np.array([0, 0, 0, 1, 1]), "y_pred": np.array([0, 0, 0, 1, 0]), "metrics": {"accuracy": 0.8, "precision": 0.78, "recall": 0.8, "f1": 0.77}},
        ],
    }

    mock_voting_result = {
        "y_true": np.concatenate([fold["y_true"] for fold in mock_evaluation_results["RandomForest"]]).tolist(),
        "y_pred": np.concatenate([fold["y_pred"] for fold in mock_evaluation_results["RandomForest"]]).tolist(),
    }

    save_voting_result(mock_voting_result)
    display_all_confusion_matrices(mock_evaluation_results)

if __name__ == "__main__":
    test_mock_voting()