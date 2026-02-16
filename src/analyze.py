from src.data_loader import download_data, load_data

def main():
    path = download_data()
    df = load_data(path)
    print(df.head())
    print(df.shape)
    print(df.info())
    print(f"{df['type'].value_counts(normalize=True) * 100}")

if __name__ == "__main__":
    main()
