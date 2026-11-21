import io
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.common import DATA, RAW, REPORTS, config, database, labels, output, staged, write_json


def assign_split(ids, cfg):
    games = sorted(ids.unique())
    _, validation = train_test_split(games, test_size=cfg["validation_fraction"], random_state=cfg["seed"])
    return np.where(ids.isin(validation), "val", "train")


def prepare_tabular(cfg):
    frame = pd.read_parquet(staged("tabular"))
    initial = len(frame)
    frame["primary_genre"] = frame.genres.map(labels).map(lambda items: items[0] if items else None)
    frame = frame.dropna(subset=["primary_genre"]).copy()
    frame["split"] = assign_split(frame.app_id, cfg)
    frame["title"] = frame.title.fillna("Unknown")
    for column in ["developers", "publishers", "supported_languages"]:
        values = frame[column].map(labels)
        frame[f"{column}_count"] = values.map(len)
        frame[column] = values.map(lambda items: items or ["Unknown"])
    for language in cfg["languages"]:
        key = language.lower().replace(" - ", "_").replace(" ", "_")
        frame[f"lang_{key}"] = frame.supported_languages.map(lambda items: int(language in items)).astype("uint8")
    frame["release_year"] = frame.release_date.dt.year
    numeric = ["release_year", "developers_count", "publishers_count", "supported_languages_count"]
    scaler = StandardScaler().fit(frame.loc[frame.split.eq("train"), numeric])
    frame[[f"{c}_z" for c in numeric]] = scaler.transform(frame[numeric]).astype("float32")
    with output(DATA / "processed/tabular/clean.parquet") as temp:
        frame.drop(columns=["genres"]).to_parquet(temp, index=False, compression="zstd")
    return {"input": initial, "output": len(frame), "dropped_missing_target": initial - len(frame),
            "split": frame.split.value_counts().to_dict(), "target": "primary_genre"}


def prepare_images(cfg):
    frame = pd.read_parquet(staged("images"))
    frame["split"] = assign_split(frame.app_id, cfg)
    manifest = []
    with output(DATA / "processed/images/dataset.zip") as temp, zipfile.ZipFile(temp, "w") as archive:
        for row in frame.itertuples(index=False):
            original = (RAW / cfg["image_directory"] / row.filename).read_bytes()
            name = f"{row.split}/{row.filename}"
            archive.writestr(name, original)
            manifest.append((row.app_id, name, row.genres, row.split, "original"))
            if row.split == "train":
                with Image.open(io.BytesIO(original)) as image, io.BytesIO() as buffer:
                    ImageEnhance.Brightness(image.convert("RGB")).enhance(0.85).save(buffer, "JPEG", quality=90)
                    name = f"train/augmented/{row.filename}"
                    archive.writestr(name, buffer.getvalue())
                manifest.append((row.app_id, name, row.genres, row.split, "brightness_0.85"))
        table = pd.DataFrame(manifest, columns=["app_id", "path", "genres", "split", "variant"])
        archive.writestr("manifest.csv", table.to_csv(index=False))
    return {"originals": len(frame), "augmented_train": len(table) - len(frame),
            "games_with_repaired_genre_list": int(frame.loc[frame.genre_bracket_repaired, "app_id"].nunique()),
            "original_split": frame.split.value_counts().to_dict(),
            "games_per_split": frame.groupby("split").app_id.nunique().to_dict()}


def prepare_reviews(cfg):
    quantile = float(cfg["hours_quantile"])
    if not 0 < quantile <= 1:
        raise ValueError("hours_quantile must be in (0, 1]")
    with database() as db:
        db.read_parquet(str(staged("big"))).create_view("reviews")
        count, cutoff = db.sql(f"SELECT count(*), quantile_disc(hours, {quantile}) FROM reviews").fetchone()
        cleaned = db.sql("""
            SELECT *, ln(1 + hours) AS log_hours FROM reviews WHERE hours <= $cutoff
        """, params={"cutoff": cutoff})
        with output(DATA / "processed/big/reviews.parquet") as temp:
            cleaned.write_parquet(str(temp), compression="zstd", row_group_size=122880)
            retained = db.read_parquet(str(temp)).count("*").fetchone()[0]
    return {"input": count, "output": retained, "removed_high_hours": count - retained,
            "hours_quantile": quantile, "max_hours_kept": cutoff, "target": "is_recommended"}


def prepare(dataset):
    writers = {"tabular": prepare_tabular, "images": prepare_images, "big": prepare_reviews}
    summary = writers[dataset](config(dataset))
    write_json(REPORTS / f"{dataset}_prep.json", summary)
