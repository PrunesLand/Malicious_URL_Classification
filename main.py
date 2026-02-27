import pandas as pd
from src.data_loader import download_data, load_data
from src.features import extract_url_features
from src.models import get_models
from src.evaluate import evaluate_models
from src.results import display_all_confusion_matrices, print_results, save_result

def main():
    #  Download and Load Data
    path = download_data()
    df = load_data(path)

    #  Feature Engineering
    X, y = extract_url_features(df)

    #  Label Encoding
    y_encoded = pd.factorize(y, sort=True)[0]

    #  Get Models
    models = get_models()

    #  Evaluate
    results, voting_data = evaluate_models(X, y_encoded, models)
    
    #  Print Results
    print_results(results, models)
    save_result(voting_data)
    display_all_confusion_matrices(results)
    

if __name__ == "__main__":
    main()
