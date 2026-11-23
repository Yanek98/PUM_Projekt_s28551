from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.automl.tracking import candidate


def train(path, execution):
    table = pd.read_parquet(path)
    train_rows = table.split.eq("train")
    val_rows = table.split.eq("val")
    numeric = [c for c in table if c.startswith("lang_") or c.endswith("_z")]
    title = make_pipeline(TfidfVectorizer(max_features=8000, sublinear_tf=True, dtype=np.float32),
                          TruncatedSVD(n_components=32, random_state=42), StandardScaler())
    features = ColumnTransformer([("title", title, "title"), ("numeric", "passthrough", numeric)])
    x_train = features.fit_transform(table.loc[train_rows])
    x_val = features.transform(table.loc[val_rows])
    y_train, y_val = table.primary_genre[train_rows], table.primary_genre[val_rows]
    models = {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=250),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=15, min_samples_leaf=5,
                                               max_features="sqrt", n_jobs=3, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=80, max_depth=16, min_samples_leaf=5,
                                          n_jobs=3, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=45, early_stopping=True, random_state=42),
    }
    ids = []
    for name, model in models.items():
        params = {**model.get_params(deep=False), "train_rows": len(y_train), "val_rows": len(y_val),
                  "features": "title TF-IDF/SVD32 + scaled counts/year and language flags"}
        with candidate("tabular", execution, name, params) as rid, TemporaryDirectory() as tmp:
            model.fit(x_train, y_train)
            prediction = model.predict(x_val)
            mlflow.log_metrics({"f1_weighted": f1_score(y_val, prediction, average="weighted", zero_division=0),
                                "f1_macro": f1_score(y_val, prediction, average="macro", zero_division=0),
                                "accuracy": accuracy_score(y_val, prediction)})
            target = Path(tmp) / "pipeline.joblib"
            joblib.dump(Pipeline([("features", features), ("classifier", model)]), target, compress=3)
            mlflow.log_artifact(str(target), "model")
            ids.append(rid)
    return ids
