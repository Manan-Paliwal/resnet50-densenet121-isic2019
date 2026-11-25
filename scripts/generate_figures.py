"""Build figures and shared tables from saved aggregate results.

Run from the repository root with python scripts/generate_figures.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_comparisons import discover_experiments, generate_comparisons


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
COMPARISONS = RESULTS / "Model Comparison Results"
CLASS_LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(folder):
    single_history = folder / "history.csv"
    if single_history.is_file():
        histories = [pd.read_csv(single_history)]
        stages = ["Training"]
    else:
        histories = [pd.read_csv(folder / filename) for filename in ("stage1_history.csv", "stage2_history.csv")]
        stages = ["Stage 1", "Stage 2"]

    # Use the recorded epoch order, with Stage 2 placed after Stage 1.
    epoch_axes = []
    offset = 0
    for history in histories:
        epoch_axes.append(np.arange(offset + 1, offset + len(history) + 1))
        offset += len(history)

    curves = [
        ("loss_curves.png", "Cross-entropy loss", [("train_loss","Train"),("val_loss","Validation")], 1),
        ("accuracy_curves.png", "Accuracy (%)", [("train_accuracy","Train"),("val_accuracy","Validation")], 100),
        ("validation_macro_f1.png", "Validation macro F1 (%)", [("val_macro_f1","Validation")], 100),
    ]
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout="constrained")
    for ax, (filename,ylabel,series,scale) in zip(axes,curves):
        detail, detail_ax = plt.subplots(figsize=(8,4.4),layout="constrained")
        for chart in (ax, detail_ax):
            for history, epochs, stage in zip(histories,epoch_axes,stages):
                for column,label in series:
                    chart.plot(epochs, history[column]*scale, marker="o", markersize=3, label=f"{stage} · {label}")
            if len(histories)>1:
                chart.axvline(len(histories[0])+0.5,color="grey",linestyle="--",linewidth=1)
            chart.set(xlabel="Epoch (cumulative)",ylabel=ylabel)
            if scale==100:
                chart.set_ylim(0,100)
            chart.grid(alpha=0.25)
            chart.legend(fontsize=8)
        detail_ax.set_title(folder.name)
        save_figure(detail,folder/"figures"/filename)
    fig.suptitle(folder.name)
    save_figure(fig,folder/"figures"/"training_curves.png")

    lr, ax = plt.subplots(figsize=(8,4.4),layout="constrained")
    for history, epochs, stage in zip(histories,epoch_axes,stages):
        for column in ("learning_rate","backbone_learning_rate","classifier_learning_rate","backbone_lr","classifier_lr"):
            if column in history and history[column].gt(0).any():
                ax.plot(epochs,history[column].where(history[column]>0),marker="o",markersize=3,
                        label=f"{stage} · {column.replace('_',' ')}")
    if ax.lines:
        ax.set(title=folder.name,xlabel="Epoch (cumulative)",ylabel="Recorded learning rate",yscale="log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        save_figure(lr,folder/"figures"/"learning_rates.png")
    else:
        plt.close(lr)


def plot_confusion(folder):
    matrix = pd.read_csv(folder/"confusion_matrix.csv",index_col=0).to_numpy(dtype=int)
    if matrix.shape != (8,8):
        raise ValueError(f"{folder.name}: expected an 8 × 8 confusion matrix.")

    # Keep only aggregate counts in the misclassification summary.
    errors = [
        {"true_class":CLASS_LABELS[row],"predicted_class":CLASS_LABELS[col],"count":int(matrix[row,col])}
        for row in range(8) for col in range(8) if row != col and matrix[row,col]
    ]
    pd.DataFrame(errors,columns=["true_class","predicted_class","count"]).sort_values(
        "count",ascending=False,kind="stable"
    ).to_csv(folder/"misclassification_summary.csv",index=False)

    row_totals = matrix.sum(axis=1,keepdims=True)
    normalized = np.divide(matrix,row_totals,out=np.zeros(matrix.shape,dtype=float),where=row_totals!=0)
    for filename,values,vmax,label,fmt in (
        ("confusion_matrix_raw.png",matrix,max(1,int(matrix.max())),"Image count","d"),
        ("confusion_matrix_normalized.png",normalized,1,"Fraction of true class",".2f"),
    ):
        fig,ax=plt.subplots(figsize=(7.6,6.6),layout="constrained")
        heatmap=ax.imshow(values,cmap="Blues",vmin=0,vmax=vmax)
        ax.set_xticks(range(8),CLASS_LABELS)
        ax.set_yticks(range(8),CLASS_LABELS)
        ax.set(title=folder.name,xlabel="Predicted class",ylabel="True class")
        for row in range(8):
            for col in range(8):
                value=values[row,col]
                ax.text(col,row,format(value,fmt),ha="center",va="center",fontsize=9,
                        color="white" if value>=vmax/2 else "black")
        fig.colorbar(heatmap,ax=ax,label=label)
        save_figure(fig,folder/"figures"/filename)


def plot_test_comparison(experiments):
    numbers, scores = [], []
    for number, folder in experiments:
        numbers.append(number)
        scores.append(json.loads((folder/"test_results.json").read_text()))
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.4),layout="constrained")
    for ax,key,label in zip(axes,("test_accuracy","test_macro_f1","test_weighted_f1"),
                            ("Test accuracy (%)","Test macro F1 (%)","Test weighted F1 (%)")):
        values=[100*result[key] for result in scores]
        bars=ax.bar(numbers,values,width=0.65)
        ax.set(ylabel=label,ylim=(0,100))
        ax.grid(axis="y",alpha=0.2)
        ax.bar_label(bars,fmt="%.1f%%",padding=3,fontsize=8)
    fig.suptitle("Recorded held-out performance by experiment")
    save_figure(fig,COMPARISONS/"test_metrics_comparison.png")


def main():
    experiments = discover_experiments(RESULTS)
    for _,folder in experiments:
        (folder/"figures").mkdir(exist_ok=True)
        if (folder/"history.csv").is_file() or all(
            (folder/name).is_file() for name in ("stage1_history.csv","stage2_history.csv")
        ):
            plot_training(folder)
        if (folder/"confusion_matrix.csv").is_file():
            plot_confusion(folder)
    COMPARISONS.mkdir(exist_ok=True)
    completed=[(number,folder) for number,folder in experiments if (folder/"test_results.json").is_file()]
    if completed:
        plot_test_comparison(completed)
    generate_comparisons(RESULTS)


if __name__ == "__main__":
    main()
