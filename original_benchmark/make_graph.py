"""Redraw the all-model reliability plot (seed 0) from reliability_bins.csv into reliability_redrawn.png.

Requires matplotlib (optional dependency, see requirements.txt). No models are trained.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

FOLDER = Path(__file__).resolve().parent
MODELS = ['Logistic regression', 'Random forest', 'Naive Bayes']
LINE_STYLES = [('Original', '-'), ('Sigmoid', ':')]


def main():
    bins = pd.read_csv(FOLDER / 'reliability_bins.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dataset in zip(axes, ['Breast cancer', 'Digits']):
        ax.plot([.5, 1], [.5, 1], '--', color='gray', label='Perfect calibration')
        for name in MODELS:
            for variant, style in LINE_STYLES:
                b = bins[(bins.dataset == dataset) & (bins.seed == 0) & (bins.model == name) & (bins.variant == variant)]
                ax.plot(b.confidence, b.accuracy, style, marker='o', label=f'{name}: {variant}', markersize=3)
        ax.set(title=dataset + ' (seed 0)', xlabel='Mean predicted confidence', ylabel='Fraction correct',
               xlim=(.45, 1.02), ylim=(0, 1.03))
    axes[1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    # Separate name: the redraw is not pixel-identical to the archived reliability.png.
    fig.savefig(FOLDER / 'reliability_redrawn.png', dpi=180)
    print('Saved', FOLDER / 'reliability_redrawn.png')


if __name__ == '__main__':
    main()
