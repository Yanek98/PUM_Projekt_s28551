PROCESSED = {
    "tabular": "tabular/clean.parquet",
    "images": "images/dataset.zip",
    "big": "big/reviews.parquet",
}


def artifact_uri(dataset):
    return f"file:///project/data/processed/{PROCESSED[dataset]}"
