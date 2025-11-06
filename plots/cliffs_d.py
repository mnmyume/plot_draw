from pathlib import Path
from typing import List
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from cliffs_delta import cliffs_delta

from loader.plot_config import (
    get_validation_base_dir,
    get_experiments,
    get_cliffs_d_output_path,
)
from loader.constraints_loader import ExperimentSpec as ConstraintsSpec, load_constraints_true_counts

from plots.base_plot import BasePlotter


def build_experiment_specs() -> List[ConstraintsSpec]:
    return [ConstraintsSpec(folder=e["folder"], label=e["label"]) for e in get_experiments()]


class CliffsDTablePlotter(BasePlotter):
    def __init__(self, difficulty: str | None = None, debug: bool = False) -> None:
        super().__init__()
        self.specs: List[ConstraintsSpec] = build_experiment_specs()
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

        grouped = df.groupby("experiment", sort=False, observed=False)["true_count"].apply(list)

        node_modes = grouped.index.tolist()
        n = len(node_modes)

        # Initialize empty effect size matrix
        d_matrix = np.full((n, n), np.nan)

        # Perform pairwise Cliff's Delta calculations (lower triangle only)
        for i in range(n):
            for j in range(i):
                # Row i vs Column j
                # d > 0 means row i is "greater" (more true constraints) than column j
                x, y = grouped[node_modes[i]], grouped[node_modes[j]]
                d = cliffs_delta(x, y)  # d[1]: negligible, 0.147, small, 0.33, medium, 0.474, large
                d_matrix[i, j] = d[0]

        # Prepare DataFrame
        d_df = pd.DataFrame(d_matrix, index=node_modes, columns=node_modes)

        # Interpretations for cbar, not for individual cells
        # negligible: < 0.147, small: < 0.33, medium: < 0.474, large: >= 0.474

        # Draw heatmap with text annotations
        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(d_df, dtype=bool))  # mask upper triangle
        sns.heatmap(
            d_df,
            mask=mask,
            cmap="coolwarm",  # Diverging colormap
            vmin=-1.0,  # Fix range from -1
            vmax=1.0,  # to +1
            center=0.0,  # Center at 0
            cbar_kws={"label": "Cliff's Delta (d) Effect Size"},
            annot=d_df.map(lambda p: f"{p:.2f}" if not pd.isna(p) else ""),  # Format to 2 decimal places
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
            get_cliffs_d_output_path().with_name(f"cliffs_d_{difficulty_label.lower()}.png"),
            title=f"Cliff's Delta (d) Effect Size ({difficulty_label})",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliff's Delta (d) Effect Size Table")
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
    CliffsDTablePlotter(difficulty=difficulty_filter, debug=args.debug).plot()


if __name__ == "__main__":
    main()