import great_expectations as gx
import pandas as pd

from src.common import REPORTS, config, database, labels, staged, write_json


def tabular_metrics(df, cfg):
    return {
        "rows": (len(df), cfg["min_rows"], cfg["max_rows"]),
        "column_count": (len(df.columns), len(cfg["required"]), cfg["max_columns"]),
        "duplicate_ids": (df.app_id.duplicated().sum(), 0, 0),
        "invalid_ids": ((df.app_id.isna() | df.app_id.le(0)).sum(), 0, 0),
        "missing_date": (df.release_date.isna().sum(), 0, 0),
        "missing_title_fraction": (df.title.isna().mean(), 0, cfg["max_missing_title_fraction"]),
        "unlabelled_genre_fraction": (df.genres.map(labels).map(len).eq(0).mean(), 0, cfg["max_missing_genres_fraction"]),
    }


def image_metrics(df, cfg):
    return {
        "image_count": (df.filename.count(), cfg["min_images"], None),
        "duplicate_filenames": (df.filename.dropna().duplicated().sum(), 0, 0),
        "unmatched_records": (df["match"].ne("both").sum(), 0, 0),
        "games_with_wrong_image_count": (df.groupby("app_id").filename.count().ne(cfg["images_per_game"]).sum(), 0, 0),
        "wrong_width": (df.width.ne(cfg["width"]).sum(), 0, 0),
        "wrong_height": (df.height.ne(cfg["height"]).sum(), 0, 0),
        "invalid_jpeg": (df.format.ne("JPEG").sum(), 0, 0),
        "missing_genres": (df.genre_count.fillna(0).eq(0).sum(), 0, 0),
        "missing_name": (df.name.fillna("").str.strip().eq("").sum(), 0, 0),
    }


def review_metrics(cfg):
    with database() as db:
        db.read_parquet(str(staged("big"))).create_view("reviews")
        result = db.sql("""
            SELECT count(*) AS rows,
                   count(*) - count(DISTINCT review_id) AS missing_or_duplicate_review_ids,
                   count(*) FILTER (WHERE review_id < 0 OR user_id IS NULL OR user_id < 0
                                     OR app_id IS NULL OR app_id <= 0) AS invalid_ids,
                   count(*) FILTER (WHERE hours IS NULL OR NOT isfinite(hours) OR hours < 0) AS invalid_hours,
                   count(*) FILTER (WHERE helpful IS NULL OR funny IS NULL OR least(helpful, funny) < 0) AS invalid_votes,
                   count(*) FILTER (WHERE date IS NULL) AS missing_date,
                   count(*) FILTER (WHERE is_recommended IS NULL) AS missing_target,
                   count(DISTINCT is_recommended) AS target_classes
            FROM reviews
        """)
        metrics = dict(zip(result.columns, result.fetchone()))
    checks = {name: (value, 0, 0) for name, value in metrics.items()}
    checks["rows"] = (metrics["rows"], cfg["min_rows"], None)
    checks["target_classes"] = (metrics["target_classes"], 2, 2)
    return checks


def quality(dataset):
    cfg = config(dataset)
    if dataset == "big":
        checks = review_metrics(cfg)
    else:
        frame = pd.read_parquet(staged(dataset))
        checks = (tabular_metrics if dataset == "tabular" else image_metrics)(frame, cfg)

    # Each metric summarizes the complete input, including the streamed reviews.
    metrics = pd.DataFrame([{key: value for key, (value, _, _) in checks.items()}])
    context = gx.get_context(mode="ephemeral")
    asset = context.data_sources.add_pandas(name=dataset).add_dataframe_asset(name="metrics")
    batch = asset.add_batch_definition_whole_dataframe("all").get_batch(batch_parameters={"dataframe": metrics})
    results = []
    for name, (_, low, high) in checks.items():
        expectation = gx.expectations.ExpectColumnValuesToBeBetween(column=name, min_value=low, max_value=high)
        success = bool(batch.validate(expectation).success)
        results.append(dict(metric=name, observed=metrics[name].iloc[0].item(), minimum=low, maximum=high, success=success))
    passed = all(result["success"] for result in results)
    write_json(REPORTS / f"{dataset}_quality.json", {
        "dataset": dataset, "scope": "full input", "engine": f"Great Expectations {gx.__version__}",
        "success": passed, "checks": results,
    })
    if not passed:
        raise ValueError(f"Quality failed: {[r['metric'] for r in results if not r['success']]}")
