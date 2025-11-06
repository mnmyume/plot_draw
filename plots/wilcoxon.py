from pathlib import Path
from typing import List
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon

from loader.plot_config import (
    get_validation_base_dir,
    get_experiments,
    get_wilcoxon_output_path,
)
from loader.metrics_loader import ExperimentSpec, load_metric_series
from loader.constraints_loader import ExperimentSpec as ConstraintsSpec, load_constraints_true_counts

from plots.base_plot import BasePlotter


def build_experiment_specs() -> List[ConstraintsSpec]:
    return [ConstraintsSpec(folder=e["folder"], label=e["label"]) for e in get_experiments()]


class WilcoxonTablePlotter(BasePlotter):
    def __init__(self, difficulty: str | None = None, debug: bool = False) -> None:
        super().__init__()
        self.specs: List[ExperimentSpec] = build_experiment_specs()
        self.difficulty = difficulty
        self.debug = debug

    def plot(self) -> None:
        specs_for_constraints: List[ConstraintsSpec] = [ConstraintsSpec(s.folder, s.label) for s in self.specs]
        df: pd.DataFrame = load_constraints_true_counts(
            self.base_dir, specs_for_constraints, self.difficulty, debug=self.debug
        )

        if df.empty:
            difficulty_msg = f" for difficulty '{self.difficulty}'" if self.difficulty else ""
            print(f"No constraints data found under {self.base_dir}{difficulty_msg}")
            return

        ordered_labels = [s.label for s in self.specs]

        df['experiment'] = pd.Categorical(
            df['experiment'],
            categories=ordered_labels,
            ordered=True
        )

        grouped = df.groupby("experiment", sort=False)["true_count"].apply(list)

        node_modes = grouped.index.tolist()
        n = len(node_modes)

        # Initialize empty p-value matrix
        p_matrix = np.full((n, n), np.nan)

        # Perform pairwise Wilcoxon tests (lower triangle only)
        for i in range(n):
            for j in range(i):
                x, y = grouped[node_modes[i]], grouped[node_modes[j]]
                stat, p = wilcoxon(x, y)
                p_matrix[i, j] = p

        # Prepare DataFrame
        p_df = pd.DataFrame(p_matrix, index=node_modes, columns=node_modes)

        # Add significance stars
        def starify(p):
            if pd.isna(p):
                return ""
            if p <= 0.001:
                return "***"
            elif p <= 0.01:
                return "**"
            elif p <= 0.05:
                return "*"
            else:
                return ""

        # Draw heatmap with text annotations
        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(p_df, dtype=bool))  # mask upper triangle
        sns.heatmap(
            p_df,
            mask=mask,
            cmap="coolwarm_r",
            cbar_kws={"label": "p-value"},
            annot=p_df.applymap(lambda p: f"{p:.3f}{starify(p)}" if not pd.isna(p) else ""),
            fmt="",
            linewidths=0.5,
            annot_kws={"size": 9, "color": "black"},
        )

        plt.xticks(rotation=20)
        plt.yticks(rotation=0)
        plt.xlabel("Node mode")
        plt.ylabel("Node mode")

        difficulty_label = self.difficulty.capitalize() if self.difficulty else "Overall"

        self._finalize_and_save(
            get_wilcoxon_output_path().with_name(f"wilcoxon_table_{difficulty_label.lower()}.png"),
            title=f"Wilcoxon signed-rank test results ({difficulty_label})",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wilcoxon Signed-Rank Test Result Table")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "overall"],
        default="overall",
        help="Difficulty level for constraints data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about constraint counts",
    )
    args = parser.parse_args()

    plt.close("all")
    difficulty_filter = None if args.difficulty == "overall" else args.difficulty
    WilcoxonTablePlotter(difficulty=difficulty_filter, debug=args.debug).plot()


if __name__ == "__main__":
    main()
