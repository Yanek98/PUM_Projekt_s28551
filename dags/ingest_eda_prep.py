import pendulum
from airflow.decorators import dag, task, task_group


@task
def ingest(dataset):
    from src.ingest import ingest as run
    run(dataset)


@task
def quality(dataset):
    from src.quality import quality as run
    run(dataset)


@task
def eda(dataset):
    from src.eda import profile
    profile(dataset)


@task
def prep(dataset):
    from src.prep import prepare
    prepare(dataset)


@task_group
def dataset_tasks(dataset):
    ingest(dataset) >> quality(dataset) >> eda(dataset) >> prep(dataset)


@dag(dag_id="ingest_eda_prep", start_date=pendulum.datetime(2025, 11, 1, tz="UTC"),
     schedule=None, catchup=False, max_active_runs=1, tags=["steam", "zajecia-1"])
def steam_pipeline():
    for dataset in ("tabular", "images", "big"):
        dataset_tasks.override(group_id=dataset)(dataset)


dag = steam_pipeline()
