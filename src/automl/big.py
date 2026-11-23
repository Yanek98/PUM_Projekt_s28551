from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import dask
import joblib
import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from src.automl.tracking import candidate
from src.common import database


HISTORY_END = "2021-01-01"
VALID_FROM = "2022-01-01"
SHAPE = (2, 20, 20, 6, 6, 2)  # period, game/user rate, game/user support, outcome
FEATURES = ["game_rate", "user_rate", "game_support", "user_support"]


def rate_bin(alias, prior):
    return (f"least(19, floor(20 * (coalesce({alias}.pos, 0) + 8 * CAST({prior} AS DOUBLE)) "
            f"/ (coalesce({alias}.n, 0) + 8)))::INTEGER")


def support_bin(alias):
    return f"least(5, floor(log2(1 + coalesce({alias}.n, 0))))::INTEGER"


def batch_counts(batch):
    columns = tuple(col.to_numpy(zero_copy_only=False) for col in batch.columns)
    indices = np.ravel_multi_index(columns, SHAPE)
    return np.bincount(indices, minlength=np.prod(SHAPE)).reshape(SHAPE)


def prepare_histograms(source, execution, directory):
    with database() as db, mlflow.start_run(run_name="historical-features",
            tags={"role": "feature_store", "dataset": "big", "airflow_run": execution}) as shared:
        db.read_parquet(str(source)).create_view("reviews")
        db.sql(f"SELECT * FROM reviews WHERE date < DATE '{HISTORY_END}'").create_view("history")
        prior, historical = db.sql("SELECT avg(is_recommended::INT), count(*) FROM history").fetchone()
        if not historical:
            raise ValueError("No reviews in history period")
        for key, table, minimum in (("app_id", "games", 1), ("user_id", "users", 2)):
            path = directory / f"{table}.parquet"
            db.sql(f"""SELECT {key}, count(*)::INTEGER AS n, sum(is_recommended::INT)::INTEGER AS pos
                       FROM history GROUP BY {key} HAVING count(*) >= {minimum}""").write_parquet(str(path), compression="zstd")
            db.read_parquet(str(path)).create_view(table)
            mlflow.log_artifact(str(path), "lookups")
        mlflow.log_params({"history_end_exclusive": HISTORY_END, "global_prior": prior, "history_rows": historical})
        reader = db.execute(f"""
            SELECT (r.date >= DATE '{VALID_FROM}')::INTEGER AS period,
                   {rate_bin('g', prior)} AS game_rate, {rate_bin('u', prior)} AS user_rate,
                   {support_bin('g')} AS game_support, {support_bin('u')} AS user_support,
                   r.is_recommended::INTEGER AS outcome
            FROM reviews r LEFT JOIN games g USING(app_id) LEFT JOIN users u USING(user_id)
            WHERE r.date >= DATE '{HISTORY_END}'
        """).fetch_record_batch(262144)
        counts = np.zeros(SHAPE, dtype=np.int64)
        while chunks := list(islice(reader, 4)):
            partials = dask.compute(*(dask.delayed(batch_counts)(chunk) for chunk in chunks),
                                    scheduler="threads", num_workers=2)
            counts += sum(partials)
        total = db.sql("SELECT count(*) FROM reviews").fetchone()[0]
        if int(counts.sum()) + historical != total:
            raise ValueError("Period counts do not reconcile with source")
        return counts, shared.info.run_id, historical


def weighted_rows(counts):
    game, user, ng, nu, target = counts.nonzero()
    x = np.column_stack(((game + 0.5) / 20, (user + 0.5) / 20, ng, nu))
    return x, target, counts[game, user, ng, nu, target]


class GameRateBaseline:
    def predict_proba(self, features):
        rate = features[:, 0]
        return np.column_stack((1 - rate, rate))


def train(path, execution):
    with TemporaryDirectory() as tmp:
        counts, lookup_id, historical = prepare_histograms(path, execution, Path(tmp))
    x, y, weight = weighted_rows(counts[0])
    xv, yv, wv = weighted_rows(counts[1])
    models = {
        "GamePrior": GameRateBaseline(),
        "WeightedLogistic": LogisticRegression(max_iter=300),
        "WeightedTree": DecisionTreeClassifier(max_depth=7, min_weight_fraction_leaf=0.0002, random_state=42),
        "HistogramBoost": HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=15, early_stopping=False,
                                                       min_samples_leaf=5, random_state=42),
    }
    ids = []
    for name, model in models.items():
        params = dict(history_rows=historical, train_rows=int(weight.sum()), val_rows=int(wv.sum()),
                      history_end=HISTORY_END, validation_start=VALID_FROM, lookup_run_id=lookup_id,
                      inputs=",".join(FEATURES), smooth_prior_weight=8)
        if hasattr(model, "get_params"):
            params.update(model.get_params(deep=False))
        with candidate("big", execution, name, params) as rid, TemporaryDirectory() as tmp:
            if name != "GamePrior":
                model.fit(x, y, sample_weight=weight)
            scores = model.predict_proba(xv)[:, 1]
            mlflow.log_metrics({"roc_auc": roc_auc_score(yv, scores, sample_weight=wv),
                                "f1_positive": f1_score(yv, scores >= 0.5, sample_weight=wv)})
            target = Path(tmp) / "classifier.joblib"
            joblib.dump(model, target, compress=3)
            mlflow.log_artifact(str(target), "model")
            ids.append(rid)
    return ids
