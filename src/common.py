import ast
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import duckdb
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("STEAM_RAW_DIR", ROOT / "raw_data"))
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def config(dataset):
    with (ROOT / "include" / "config" / f"{dataset}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def staged(dataset):
    return DATA / "staged" / f"{dataset}.parquet"


@contextmanager
def output(path):
    """Publish a complete file only after its writer has closed it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}-{uuid4().hex}{path.suffix}")
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, content):
    with output(path) as temp:
        temp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def database():
    scratch = DATA / "tmp"
    scratch.mkdir(exist_ok=True, parents=True)
    return duckdb.connect(config={"threads": 2, "memory_limit": "1200MB", "temp_directory": str(scratch)})


def labels(value):
    if not isinstance(value, str) or not value.strip():
        return []
    items = ast.literal_eval(value)
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError(f"Expected a list of labels: {value[:60]}")
    return [item.strip() for item in items if item.strip()]
