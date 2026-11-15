"""
Plot the cross-lingual transfer PER heatmap, with the joint multilingual model
as an extra bottom row.

Rows are the training condition (each monolingual model, then the single model
trained on all three languages' pooled data); columns are the test language.
Reads outputs/cross_lingual_results.csv and outputs/multilingual_results.csv
(both from CrossLingualEval.py), so run that first. Writes figures/cross_lingual_per_heatmap.png.

Needs matplotlib, so run with the expat library-path prefix (see CLAUDE.md):
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/python src/PlotCrossLingualHeatmap.py
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "NeuralNetwork"))  # NeuralG2P & co. live there
from NeuralG2P import ROOT, LANGUAGES

CROSS_CSV = ROOT / "outputs" / "cross_lingual_results.csv"
MULTI_CSV = ROOT / "outputs" / "multilingual_results.csv"
OUT_PNG = ROOT / "figures" / "cross_lingual_per_heatmap.png"


def read_per(path):
    """{(trained_on, tested_on): test_per} from a results CSV."""
    with open(path, newline="") as f:
        return {(r["trained_on"], r["tested_on"]): float(r["test_per"])
                for r in csv.DictReader(f)}


def main():
    per = read_per(CROSS_CSV) | read_per(MULTI_CSV)
    rows = LANGUAGES + ["multilingual"]
    grid = np.array([[per[(r, c)] for c in LANGUAGES] for r in rows])

    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=150)
    vmax = grid.max()
    im = ax.imshow(grid, cmap="viridis_r", vmin=0.0, vmax=vmax, aspect="auto")

    for i, r in enumerate(rows):
        for j, c in enumerate(LANGUAGES):
            v = grid[i, j]
            # In-domain cells (each monolingual model on its own language, and
            # every cell of the joint row) are bold.
            in_domain = r == c or r == "multilingual"
            ax.text(j, i, f"{v * 100:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold" if in_domain else "normal",
                    color="white" if v > vmax * 0.5 else "black")

    ax.set_xticks(range(len(LANGUAGES)), [c.capitalize() for c in LANGUAGES])
    ax.set_yticks(range(len(rows)),
                  [r.capitalize() for r in LANGUAGES] + ["All three\n(joint)"])
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Trained on")
    ax.set_title("Cross-lingual G2P transfer: phoneme error rate")
    fig.colorbar(im, ax=ax, label="Phoneme error rate")
    fig.tight_layout()

    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
