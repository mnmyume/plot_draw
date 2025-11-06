from pathlib import Path
from typing import List, Dict


def get_validation_base_dir() -> Path:
    return Path("data/metrics/validation").resolve()


def get_experiments() -> List[Dict[str, str]]:
    # Edit this list to control which folders are plotted and their display labels
    return [
        {"folder": "separate", "label": "separate"},
        {"folder": "merge_attra_accom", "label": "merge_attra_accom"},
        {"folder": "merge_attra_resta", "label": "merge_attra_resta"},
        {"folder": "merge_accom_resta", "label": "merge_accom_resta"},
        {"folder": "merge_all", "label": "merge_all"},
    ]

def get_token_key() -> str:
    return "completion_tokens"

def get_metric_key() -> str:
    # Change this to plot a different metric key from metrics_results.json
    return "wall_time_sec"


def get_boxplot_output_path() -> Path:
    # Output image path for constraints plots
    return Path("plots/image/boxplots/boxplot.png").resolve()


def get_wilcoxon_output_path() -> Path:
    # Output image path
    return Path("plots/image/wilcoxon/wilcoxon.png").resolve()

def get_cliffs_d_output_path() -> Path:
    return Path("plots/image/cliffs_d/cliffs_d.png").resolve()

