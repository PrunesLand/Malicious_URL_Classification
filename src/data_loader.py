import kagglehub
import pandas as pd
import os
from .config import DATASET_NAME, CSV_FILENAME

def download_data():
    path = kagglehub.dataset_download(DATASET_NAME)
    print("Path to dataset files:", path)
    print(os.listdir(path))
    return path

def load_data(path):
    df = pd.read_csv(os.path.join(path, CSV_FILENAME))
    return df
