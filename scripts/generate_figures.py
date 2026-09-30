"""Generate publication-friendly figures from committed experimental results.

Run from the repository root:
    python scripts/generate_figures.py

This reads only CSV/JSON results and writes PNG images into figures/.
It does not load data images, checkpoints, or retrain neural networks.
"""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
EXPERIMENTS = [
    {
        "title": "ResNet50 — trained from scratch",
        "slug": "resnet50_scratch",
        "histories": ["resnet50_scratch_history.csv"],
        "matrix": "resnet50_scratch_confusion_matrix.csv",
    },
    {
        "title": "DenseNet121 — trained from scratch",
        "slug": "densenet121_scratch",
        "histories": ["densenet121_scratch_history.csv"],
        "matrix": "densenet121_scratch_confusion_matrix.csv",
    },
    {
        "title": "ResNet50 — transfer learning + fine-tuning",
        "slug": "resnet50_transfer",
        "histories": [
            "resnet50_transfer_stage1_history.csv",
            "resnet50_transfer_stage2_history.csv",
        ],
        "matrix": "resnet50_transfer_confusion_matrix.csv",
    },
    {
        "title": "DenseNet121 — transfer learning + fine-tuning",
        "slug": "densenet121_transfer",
        "histories": [
            "densenet121_transfer_stage1_history.csv",
            "densenet121_transfer_stage2_history.csv",
        ],
        "matrix": None,  # No published DenseNet121 transfer test results yet.
    },
]


def read_history(name):
    df = pd.read_csv(RESULTS / name)
    required = {
        "epoch", "train_loss", "val_loss",
        "train_accuracy", "val_accuracy", "val_macro_f1",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    return df


def plot_training(experiment):
    histories = [read_history(name) for name in experiment["histories"]]

    # Preserve the chronological order of stages: feature extraction,
    # then fine-tuning. Epoch numbering restarts inside Stage 2.
    x_parts = []
    offset = 0
    for history in histories:
        xs = np.arange(1 + offset, len(history) + offset + 1)
        x_parts.append(xs)
        offset += len(history)

    stages = (["Training"] if len(histories) == 1
              else ["Feature extraction", "Fine-tuning"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), layout="constrained")
    curves = [
        ("Loss", "train_loss", "val_loss", 1.0),
        ("Accuracy (%)", "train_accuracy", "val_accuracy", 100.0),
    ]

    for ax, (title, train_key, val_key, factor) in zip(axes[:2], curves):
        for history, xs, stage in zip(histories, x_parts, stages):
            ax.plot(xs, history[train_key] * factor,
                    marker="o", markersize=3, label=f"{stage} — train")
            ax.plot(xs, history[val_key] * factor,
                    marker="s", markersize=3, label=f"{stage} — validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch (cumulative across stages)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    ax = axes[2]
    for history, xs, stage in zip(histories, x_parts, stages):
        ax.plot(xs, history["val_macro_f1"], marker="o", markersize=3,
                label=f"{stage} — validation")
    ax.set_title("Validation macro F1")
    ax.set_xlabel("Epoch (cumulative across stages)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    if len(histories) > 1:
        boundary = len(histories[0]) + 0.5
        for axis in axes:
            axis.axvline(boundary, color="gray", linestyle="--",
                         linewidth=1, alpha=0.8)

    fig.suptitle(experiment["title"] + " — recorded training history",
                 fontsize=13)
    outfile = FIGURES / f'{experiment["slug"]}_training.png'
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_confusion(experiment):
    matrix_file = experiment["matrix"]
    if matrix_file is None:
        return None

    matrix = pd.read_csv(RESULTS / matrix_file, index_col=0)
    matrix_values = matrix.to_numpy(dtype=float)

    if matrix_values.shape != (8, 8):
        raise ValueError(
            f"{matrix_file}: expected 8 x 8 confusion matrix, "
            f"got {matrix_values.shape}"
        )

    row_totals = matrix_values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix_values, row_totals, out=np.zeros_like(matrix_values),
        where=row_totals != 0
    )

    fig, ax = plt.subplots(figsize=(9, 7.5), layout="constrained")
    img = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8), CLASS_LABELS)
    ax.set_yticks(range(8), CLASS_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        experiment["title"] + "\n"
        "Normalized test confusion matrix (row-wise)"
    )

    for row in range(8):
        for col in range(8):
            val = normalized[row, col]
            ax.text(col, row, f"{val:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if val >= 0.5 else "black")

    fig.colorbar(img, ax=ax, shrink=0.75, label="Fraction of actual class")
    outfile = FIGURES / f'{experiment["slug"]}_confusion_matrix.png'
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_test_comparison():
    sources = [
        ("ResNet50\nscratch", "resnet50_scratch_test_results.json"),
        ("DenseNet121\nscratch", "densenet121_scratch_test_results.json"),
        ("ResNet50\ntransfer", "resnet50_transfer_test_results.json"),
    ]
    names, accuracy, macro_f1 = [], [], []
    for label, filename in sources:
        data = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
        names.append(label)
        accuracy.append(100 * data["test_accuracy"])
        macro_f1.append(data["test_macro_f1"])

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             layout="constrained")
    for ax, values, label, ylim, fmt in [
        (axes[0], accuracy, "Test accuracy (%)", (0, 100), "{:.1f}%"),
        (axes[1], macro_f1, "Test macro F1", (0, 1), "{:.3f}"),
    ]:
        bars = ax.bar(x, values, width=0.65)
        ax.set_xticks(x, names)
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.annotate(fmt.format(value),
                        (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle("Reported held-out test metrics (n=3,808 per run)")
    outfile = FIGURES / "test_metrics_comparison.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main():
    outputs = []
    for experiment in EXPERIMENTS:
        outputs.append(plot_training(experiment))
        cm = plot_confusion(experiment)
        if cm is not None:
            outputs.append(cm)
    outputs.append(plot_test_comparison())

    print(f"Generated {len(outputs)} figures:")
    for output in outputs:
        print(" -", output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
