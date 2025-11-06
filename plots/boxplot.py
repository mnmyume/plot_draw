from pathlib import Path
from typing import List

import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from loader.plot_config import (
    get_validation_base_dir,
    get_experiments,
    get_metric_key,
    get_token_key,
    get_boxplot_output_path,
)
from loader.metrics_loader import ExperimentSpec, load_metric_series, load_token_series
from loader.constraints_loader import ExperimentSpec as ConstraintsSpec, load_constraints_true_counts

from plots.base_plot import BasePlotter


def build_experiment_specs() -> List[ExperimentSpec]:
    return [ExperimentSpec(folder=e["folder"], label=e["label"]) for e in get_experiments()]


class MetricsBoxPlotter(BasePlotter):
    def __init__(self, difficulty: str | None = None, debug: bool = False) -> None:
        super().__init__()
        self.specs: List[ExperimentSpec] = build_experiment_specs()
        self.difficulty = difficulty
        self.debug = debug

    def plot(self) -> None:
        metric_key: str = get_metric_key()
        df: pd.DataFrame = load_metric_series(self.base_dir, self.specs, metric_key)
        if df.empty:
            print(f"No data found for metric '{metric_key}' under {self.base_dir}")
            return
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df,
            x="experiment",
            y=metric_key,
            showmeans=True,
            meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": "8"}
        )

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        ax.set_xlabel("Node Mode")
        ax.set_ylabel(metric_key)
        ax.set_title(f"Validation {metric_key} across experiments")
        self._finalize_and_save(
            get_boxplot_output_path().with_name(f"boxplot_metrics_{metric_key}.png"),
            title=f"{metric_key}",
        )


class TokenBoxPlotter(BasePlotter):
    def __init__(self, difficulty: str | None = None, debug: bool = False) -> None:
        super().__init__()
        self.specs: List[ExperimentSpec] = build_experiment_specs()
        self.difficulty = difficulty
        self.debug = debug

    def plot(self) -> None:
        token_key: str = get_token_key()
        df: pd.DataFrame = load_token_series(self.base_dir, self.specs, token_key)
        if df.empty:
            print(f"No data found for token '{token_key}' under {self.base_dir}")
            return
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df,
            x="experiment",
            y=token_key,
            showmeans=True,
            meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": "8"}
        )

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        ax.set_xlabel("Node Mode")
        ax.set_ylabel(token_key)
        ax.set_title(f"Validation {token_key} across experiments")
        self._finalize_and_save(
            get_boxplot_output_path().with_name(f"boxplot_metrics_{token_key}.png"),
            title=f"{token_key}",
        )


class ConstraintsBoxPlotter(BasePlotter):
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

        if self.debug:
            print(f"\nOverall statistics:")
            print(df.groupby("experiment")["true_count"].describe())

        plt.figure(figsize=(10, 6))

        ax = sns.boxplot(
            data=df,
            x="experiment",
            y="true_count",
            showmeans=True,
            meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": "8"}
        )

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        ax.set_xlabel("Experiment")
        ax.set_ylabel("# True constraints per query")
        difficulty_label = self.difficulty.capitalize() if self.difficulty else "Overall"
        ax.set_title(f"Per-query constraints satisfaction across experiments ({difficulty_label})")
        self._finalize_and_save(
            get_boxplot_output_path().with_name(f"boxplot_constraints_{difficulty_label.lower()}.png"),
            title=f"Per-query constraints satisfaction ({difficulty_label})",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation boxplots")
    parser.add_argument(
        "--plot",
        choices=["metrics", "constraints", "tokens"],
        default="metrics",
        help="Which plot to generate",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "overall"],
        default="overall",
        help="Difficulty level for constraints plot (only used with --plot constraints)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about constraint counts",
    )
    args = parser.parse_args()

    plt.close('all')

    if args.plot in ("metrics"):
        print("Generating metrics plot...")
        MetricsBoxPlotter().plot()
    if args.plot in ("tokens"):
        print("Generating tokens plot...")
        TokenBoxPlotter().plot()
    if args.plot in ("constraints"):
        print("Generating constraints plot...")
        difficulty_filter = None if args.difficulty == "overall" else args.difficulty
        ConstraintsBoxPlotter(difficulty=difficulty_filter, debug=args.debug).plot()


if __name__ == "__main__":
    main()