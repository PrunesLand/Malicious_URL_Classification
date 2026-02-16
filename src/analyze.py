from src.data_loader import download_data, load_data
from src.features import extract_url_features
import pandas as pd

def main():
    path = download_data()
    df = load_data(path)
    print(df.head())
    print(df.shape)
    print(df.info())
    print(f"{df['type'].value_counts(normalize=True) * 100}")

    X, y = extract_url_features(df)
    
    combined_df = pd.concat([X, y], axis=1)
    print(combined_df.head())
    

if __name__ == "__main__":
    main()
