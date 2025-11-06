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


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _count_true_in_query(query_obj: Dict[str, Any]) -> int:
    total_true = 0
    # query_obj contains sections like "Commonsense Constraint" and "Hard Constraint"
    for section in query_obj.values():
        if isinstance(section, dict):
            for v in section.values():
                if v is True:
                    total_true += 1
    return total_true


def load_constraints_true_counts(
    base_dir: Path,
    experiments: Iterable[ExperimentSpec],
    difficulty_filter: str | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Load constraints true counts from per_query_constraints.json files.
    
    Args:
        base_dir: Base directory containing experiment folders
        experiments: List of experiment specifications
        difficulty_filter: One of "easy", "medium", "hard", or None for "overall"
        debug: If True, print debug information about counts
    
    Returns:
        DataFrame with columns "experiment" and "true_count"
    """
    records: List[Dict[str, Any]] = []
    valid_difficulties = {"easy", "medium", "hard"}
    
    if difficulty_filter is not None and difficulty_filter not in valid_difficulties:
        raise ValueError(f"difficulty_filter must be one of {valid_difficulties} or None")
    
    for exp in experiments:
        json_path = base_dir / exp.folder / "per_query_constraints.json"
        data = _read_json(json_path)
        if not isinstance(data, dict):
            continue

        # Structure: difficulty -> num_categories (e.g., "3","5","7") -> query_id -> sections
        difficulties_to_process = [difficulty_filter] if difficulty_filter else data.keys()
        
        query_count = 0
        for difficulty_key in difficulties_to_process:
            if difficulty_key not in data:
                continue
            difficulty_obj = data[difficulty_key]
            if not isinstance(difficulty_obj, dict):
                continue
            for group_obj in difficulty_obj.values():
                if not isinstance(group_obj, dict):
                    continue
                for _query_id, query_obj in group_obj.items():
                    if isinstance(query_obj, dict):
                        count_true = _count_true_in_query(query_obj)
                        records.append({
                            "experiment": exp.label,
                            "true_count": count_true,
                        })
                        query_count += 1
        
        if debug:
            counts = [r["true_count"] for r in records if r["experiment"] == exp.label]
            if counts:
                print(f"{exp.label}: {len(counts)} queries, mean={sum(counts)/len(counts):.2f}, "
                      f"min={min(counts)}, max={max(counts)}, unique={len(set(counts))}")

    if not records:
        return pd.DataFrame(columns=["experiment", "true_count"])
    return pd.DataFrame.from_records(records)


