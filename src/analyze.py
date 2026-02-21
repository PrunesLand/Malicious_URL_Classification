from src.data_loader import download_data, load_data
from src.features import extract_url_features
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    counts = df['type'].value_counts(normalize=True)
    print(f"{counts * 100}")
    
    plt.figure(figsize=(8, 8))
    counts.plot.pie(colors=plt.cm.Pastel1.colors, autopct='%1.1f%%')
    plt.title('Class Distribution of URLs')
    plt.ylabel('')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/class_distribution.png')

    print("\n === Post Feature Engineering ===\n")
    X, y = extract_url_features(df)
    
    combined_df = pd.concat([X, y], axis=1)
    print(combined_df.info())
    

if __name__ == "__main__":
    main()
