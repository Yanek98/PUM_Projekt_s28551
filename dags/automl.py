import pendulum
from airflow.datasets import Dataset
from airflow.decorators import dag, task
from src.assets import PROCESSED, artifact_uri


@dag(start_date=pendulum.datetime(2025, 11, 1, tz="UTC"), schedule=[Dataset(artifact_uri(k)) for k in PROCESSED],
     catchup=False, max_active_runs=1, tags=["steam", "zajecia-2"])
def automl():
    @task
    def check_inputs():
        from src.automl.tracking import check_inputs
        check_inputs()

    @task
    def train(dataset, dag_run=None):
        from src.automl.tracking import train_dataset
        return train_dataset(dataset, dag_run.run_id)

    @task
    def publish(candidates, dag_run=None):
        from src.automl.tracking import publish_results
        publish_results(candidates, dag_run.run_id)

    ready = check_inputs()
    results = []
    for dataset in ("images", "big", "tabular"):
        runs = train.override(task_id=f"train_{dataset}")(dataset)
        ready >> runs
        results.append(runs)
    publish(results)


dag = automl()
