# Malicious URL Classifier

This project classifies URLs as malicious or benign using machine learning models (XGBoost, KNN, MLP).

## Prerequisites

- Docker installed on your machine.

## How to Run with Docker

1.  **Build the Docker image:**

    Open your terminal in the project directory and run:

    ```bash
    docker build -t malicious-url-classifier .
    ```

2.  **Run the container:**

    Once the build is complete, run the application:

    ```bash
    docker run --rm malicious-url-classifier
    ```

    The `--rm` flag automatically removes the container after it finishes running to save space.

3.  **Run the analysis script:**

    To view the nature of the dataset using the cache:

    ```bash
    docker run --rm -v $(pwd)/kaggle_cache:/root/.cache/kagglehub malicious-url-classifier python -m src.analyze
    ```

## Notes

- The script uses `kagglehub` to download the dataset. If the dataset requires authentication, you may need to pass your Kaggle credentials as environment variables:
    ```bash
    docker run --rm -e KAGGLE_USERNAME=your_username -e KAGGLE_KEY=your_key malicious-url-classifier
    ```
- The dataset is downloaded inside the container. To persist the dataset between runs (to avoid re-downloading), you can mount a volume:
    ```bash
    docker run --rm -v $(pwd)/kaggle_cache:/root/.cache/kagglehub malicious-url-classifier
    ```

## running Locally

To run the analysis script locally without Docker, execute the following from the project root:

```bash
pip install -r requirements.txt
python3 -m src.analyze
```
