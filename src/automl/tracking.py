import math
import zipfile
from contextlib import contextmanager
from importlib import import_module

import mlflow
import pyarrow.parquet as pq
from mlflow import MlflowClient

from src.assets import PROCESSED
from src.common import DATA, REPORTS, ROOT, output, write_json


INPUTS = {key: DATA / "processed" / path for key, path in PROCESSED.items()}
METRICS = {"tabular": "f1_weighted", "images": "f1_micro", "big": "roc_auc"}


def check_inputs():
    for dataset, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Run ingest_eda_prep first. Missing {path}")
        if dataset == "images":
            with zipfile.ZipFile(path) as archive:
                archive.getinfo("manifest.csv")
        elif pq.ParquetFile(path).metadata.num_rows == 0:
            raise ValueError(f"Empty stage 1 artifact: {path}")


@contextmanager
def candidate(dataset, execution, name, params):
    with mlflow.start_run(run_name=name, tags={"airflow_run": execution, "dataset": dataset,
                                               "role": "candidate", "candidate": name}) as run:
        mlflow.log_params(params)
        yield run.info.run_id


def train_dataset(dataset, execution):
    mlflow.set_experiment(f"Steam-{dataset}")
    module = import_module(f"src.automl.{dataset}")
    return dataset, module.train(INPUTS[dataset], execution)


def export_experiment(dataset, rows, client):
    """Bundle this execution's runs and the shared features needed for inference."""
    run_ids = {row["run_id"] for row in rows}
    if dataset == "big":
        run_ids.update(row["params"]["lookup_run_id"] for row in rows)
    store = ROOT / "mlruns"
    experiment = store / client.get_run(rows[0]["run_id"]).info.experiment_id
    with output(REPORTS / "mlflow" / f"{dataset}.zip") as temp:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            meta = experiment / "meta.yaml"
            archive.write(meta, meta.relative_to(store))
            for rid in sorted(run_ids):
                for file in sorted((experiment / rid).rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(store))


def publish_results(candidates, execution):
    client = MlflowClient()
    rankings = {}
    for dataset, ids in candidates:
        metric = METRICS[dataset]
        rows = []
        for rid in ids:
            run = client.get_run(rid)
            score = run.data.metrics[metric]
            if (run.info.status != "FINISHED" or run.data.tags["airflow_run"] != execution
                    or not math.isfinite(score) or not client.list_artifacts(rid, "model")):
                raise ValueError(f"Incomplete MLflow result: {rid}")
            rows.append({"model": run.data.tags["candidate"], "run_id": rid,
                         "metrics": run.data.metrics, "params": run.data.params})
        if len({row["model"] for row in rows}) < 3:
            raise ValueError(f"Insufficient candidates for {dataset}")
        rankings[dataset] = sorted(rows, key=lambda row: (-row["metrics"][metric], row["model"]))
        export_experiment(dataset, rows, client)
    lines = ["# Steam · wyniki AutoML", "", f"Przebieg Airflow `{execution}`. Wyniki na walidacji.", ""]
    for dataset, title in [("tabular", "Katalog gier"), ("images", "Zrzuty ekranu"), ("big", "Recenzje")]:
        metric = METRICS[dataset]
        lines += [f"## {title}", "", f"| Miejsce | Model | {metric} | MLflow run ID |",
                  "|---:|---|---:|---|"]
        for rank, row in enumerate(rankings[dataset][:3], 1):
            lines.append(f"| {rank} | {row['model']} | {row['metrics'][metric]:.6f} | `{row['run_id']}` |")
        lines.append("")
    lines.append("Wszystkie porównane modele, parametry i metryki zawiera [zestawienie JSON](automl_runs.json).")
    lines += ["", "Archiwa MLflow z wagami i metrykami: [katalog](mlflow/tabular.zip), "
              "[obrazy](mlflow/images.zip), [recenzje](mlflow/big.zip)."]
    write_json(REPORTS / "automl_runs.json", {"airflow_run": execution, "selection": METRICS, "candidates": rankings})
    with output(REPORTS / "models_results.md") as path:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
