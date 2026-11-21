import pandas as pd
from ydata_profiling import ProfileReport

from src.common import REPORTS, config, database, labels, output, staged


def profile(dataset):
    if dataset == "big":
        size = int(config(dataset)["eda_sample"])
        with database() as db:
            db.read_parquet(str(staged(dataset))).create_view("reviews")
            frame = db.sql(f"SELECT * FROM reviews ORDER BY hash(review_id), review_id LIMIT {size}").df()
        scope = f"Próba {len(frame):,} recenzji według hasza review_id. Kontrole jakości obejmują cały zbiór."
    elif dataset == "images":
        frame = pd.read_parquet(staged(dataset))
        frame = frame[["app_id", "name", "genre_count", "width", "height", "format", "genres"]].copy()
        frame["primary_genre"] = frame.pop("genres").map(lambda value: labels(value)[0])
        scope = "Wszystkie oryginalne zrzuty. Jeden wiersz opisuje jeden obraz."
    else:
        frame = pd.read_parquet(staged(dataset))
        for column in ["developers", "publishers", "supported_languages", "genres"]:
            values = frame.pop(column).map(labels)
            frame[f"{column}_count"] = values.map(len)
            if column == "genres":
                frame["primary_genre"] = values.map(lambda items: items[0] if items else None)
        scope = "Pełny katalog Steam. Pola wielowartościowe podsumowano liczbą etykiet."
    with output(REPORTS / f"{dataset}_eda.html") as temp:
        ProfileReport(frame, title=f"Steam · {dataset} · EDA", dataset={"description": scope},
                      minimal=True, progress_bar=False, pool_size=1).to_file(temp)
