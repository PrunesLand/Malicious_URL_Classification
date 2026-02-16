import pandas as pd
from src.data_loader import download_data, load_data
from src.features import extract_url_features, get_target_encoder
from src.models import get_models
from src.evaluate import evaluate_models, print_results

def main():
    #  Download and Load Data
    path = download_data()
    df = load_data(path)

    #  Feature Engineering
    X, y = extract_url_features(df)

    #  Target Encoding
    le = get_target_encoder()
    y_encoded = le.fit_transform(y)

    #  Get Models
    models = get_models()

    #  Evaluate
    results = evaluate_models(X, y_encoded, models)
    
    #  Print Results
    print_results(results, models)

if __name__ == "__main__":
    main()
