from src.data_loader import download_data, load_data
from src.features import extract_url_features
import pandas as pd

def main():
    path = download_data()
    df = load_data(path)
    print("\n === Head ===\n")
    print(df.head())
    print("\n === Shape ===\n")
    print(df.shape)
    print("\n === Info ===\n")
    print(df.info())
    print("\n === Value Counts ===\n")
    print(f"{df['type'].value_counts(normalize=True) * 100}")
    print("\n === Post Feature Engineering ===\n")
    X, y = extract_url_features(df)
    
    combined_df = pd.concat([X, y], axis=1)
    print(combined_df.info())
    

if __name__ == "__main__":
    main()
