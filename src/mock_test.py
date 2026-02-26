import numpy as np
from src.results import save_voting_result, display_confusion_matrix

def test_mock_voting():
    mock_data = [
        {
            "fold": 1,
            "y_true": np.array([0, 1, 0, 1, 1]),
            "y_pred": np.array([0, 1, 1, 1, 0]) 
        },
        {
            "fold": 2,
            "y_true": np.array([1, 0, 1, 1, 0]),
            "y_pred": np.array([1, 0, 1, 0, 0]) 
        },
        {
            "fold": 3,
            "y_true": np.array([0, 0, 0, 1, 1]),
            "y_pred": np.array([0, 0, 0, 1, 1]) 
        }
    ]

    save_voting_result(mock_data)
    display_confusion_matrix()

if __name__ == "__main__":
    test_mock_voting()