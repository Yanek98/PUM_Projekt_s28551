import csv
import re

import pandas as pd
from PIL import Image

from src.common import RAW, config, database, labels, output, staged


REVIEW_TYPES = {
    "app_id": "BIGINT", "helpful": "INTEGER", "funny": "INTEGER", "date": "DATE",
    "is_recommended": "BOOLEAN", "hours": "DOUBLE", "user_id": "BIGINT", "review_id": "BIGINT",
}


def image_inventory(path, cfg):
    metadata = pd.read_csv(path, usecols=cfg["required"], keep_default_na=False)
    # Some source lists end after the fifth quoted label, without the closing bracket.
    genre_text = metadata.genres.str.strip()
    missing_bracket = genre_text.str.startswith("[") & ~genre_text.str.endswith("]")
    metadata["genre_bracket_repaired"] = missing_bracket
    metadata["genres"] = genre_text.mask(missing_bracket, genre_text + "]")
    metadata["genre_count"] = metadata.genres.map(labels).map(len)
    entries = []
    for image_path in sorted((RAW / cfg["image_directory"]).iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = re.fullmatch(r"(\d+)_(\d+)", image_path.stem)
        if not match:
            raise ValueError(f"Cannot match image to Steam ID: {image_path.name}")
        width, height, fmt = 0, 0, "unreadable"
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                fmt = image.format
        except (OSError, ValueError):
            pass  # Remains in the inventory and fails the image quality suite.
        entries.append((int(match[1]), image_path.name, width, height, fmt))
    files = pd.DataFrame(entries, columns=["appid", "filename", "width", "height", "format"])
    return metadata.merge(files, how="outer", on="appid", validate="one_to_many", indicator="match").rename(
        columns={"appid": "app_id"}
    )


def ingest(dataset):
    cfg = config(dataset)
    path = RAW / cfg["source"]
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    if len(header) != len(set(header)) or not set(cfg["required"]).issubset(header):
        raise ValueError(f"{dataset}: invalid CSV header {header}")
    with output(staged(dataset)) as temp:
        if dataset == "big":
            with database() as db:
                db.read_csv(str(path), dtype=REVIEW_TYPES, header=True).write_parquet(str(temp), compression="zstd")
        elif dataset == "images":
            image_inventory(path, cfg).to_parquet(temp, index=False)
        else:
            if len(header) > cfg["max_columns"]:
                raise ValueError(f"Too many columns in {path.name}")
            frame = pd.read_csv(path, dtype={"gameid": "Int64"}, keep_default_na=False, na_values=[""])
            frame["release_date"] = pd.to_datetime(frame["release_date"], format="%Y-%m-%d", errors="raise")
            frame.rename(columns={"gameid": "app_id"}).to_parquet(temp, index=False)
