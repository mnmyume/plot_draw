from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd


@dataclass
class ExperimentSpec:
    folder: str
    label: str


def read_metrics_json(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return data


def load_metric_series(
    base_dir: Path,
    experiments: Iterable[ExperimentSpec],
    metric_key: str,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for exp in experiments:
        exp_dir = base_dir / exp.folder
        json_path = exp_dir / "metrics_results.json"
        items = read_metrics_json(json_path)
        for item in items:
            if metric_key in item:
                value = item[metric_key]
                # Guard against non-numeric entries
                if isinstance(value, (int, float)):
                    records.append({
                        "experiment": exp.label,
                        metric_key: float(value),
                    })
    if not records:
        return pd.DataFrame(columns=["experiment", metric_key])
    return pd.DataFrame.from_records(records)
