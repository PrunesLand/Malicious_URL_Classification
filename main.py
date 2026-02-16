import pandas as pd
from src.data_loader import download_data, load_data
from src.features import extract_url_features, get_target_encoder
from src.models import get_models
from src.evaluate import evaluate_models, print_results

def main():
    # 1. Download and Load Data
    path = download_data()
    df = load_data(path)
    
    print(df.head())
    print(df.shape)
    print(df.info())
    print(f"{df['type'].value_counts(normalize=True) * 100}")

    # 2. Feature Engineering
    df = extract_url_features(df)
    
    # 3. Prepare X and y
    X = df[['url', 'url_len', 'dot_count', 'digit_count']]
    y = df['type']

    # 4. Target Encoding
    le = get_target_encoder()
    y_encoded = le.fit_transform(y)

    # 5. Get Models
    models = get_models()

    # 6. Evaluate
    results = evaluate_models(X, y_encoded, models)
    
    # 7. Print Results
    print_results(results, models)

if __name__ == "__main__":
    main()
