from src.data_loader import download_data, load_data

def main():
    path = download_data()
    df = load_data(path)
    print(df.head())

if __name__ == "__main__":
    main()
