from pathlib import Path
from typing import List
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


from loader.plot_config import (
    get_validation_base_dir,
)

class BasePlotter:
    def __init__(self) -> None:
        sns.set_theme(style="ticks", palette="pastel")
        self.base_dir: Path = get_validation_base_dir()

    def _finalize_and_save(self, output_path: Path, title: str) -> None:
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved Wilcoxon result table to {output_path}")
        plt.close()