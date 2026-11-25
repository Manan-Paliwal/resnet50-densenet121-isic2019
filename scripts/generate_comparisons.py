"""Create aggregate comparison files from the recorded experiment summaries."""

import csv
import json
from math import isfinite
from pathlib import Path


RECIPES = {
    "01": "From scratch",
    "02": "From scratch",
    "03": "Transfer learning",
    "04": "Transfer learning",
    "05": "Transfer learning + augmentation",
    "06": "Transfer learning + augmentation",
}

COMPARISONS = {
    "architecture_comparison.csv": [
        ("From scratch", "01", "02"),
        ("Transfer learning", "03", "04"),
        ("Transfer learning + augmentation", "05", "06"),
    ],
    "transfer_comparison.csv": [
        ("ResNet50", "01", "03"),
        ("DenseNet121", "02", "04"),
    ],
    "augmentation_comparison.csv": [
        ("ResNet50", "03", "05"),
        ("DenseNet121", "04", "06"),
    ],
}

SCORE_FIELDS = (
    "best_validation_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
)
OVERALL_FIELDS = (
    "experiment",
    "architecture",
    "training_recipe",
    "result_folder",
    "selected_stage",
    "selected_epoch",
    *SCORE_FIELDS,
    "test_macro_precision",
    "test_macro_recall",
    "test_loss",
    "test_samples",
    "training_minutes",
    "total_parameters",
    "training_hardware",
    "training_precision",
)


def discover_experiments(results_root: Path):
    """Find directories named '<architecture> - <number> Results', sorted by number."""
    experiments = {}
    for folder in results_root.iterdir():
        if not folder.is_dir() or not folder.name.endswith(" Results"):
            continue
        architecture, separator, number = folder.name.removesuffix(" Results").rpartition(" - ")
        if not separator or not architecture or not number.isascii() or not number.isdecimal():
            continue
        experiment = f"{int(number):02d}"
        if experiment in experiments:
            raise ValueError(f"More than one results folder for experiment {experiment}.")
        experiments[experiment] = folder
    return sorted(experiments.items(), key=lambda entry: int(entry[0]))


def first_recorded(summary, *keys):
    """Return the first available value across earlier and later summary formats."""
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def finite_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def write_table(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_paired_macro_f1(comparison_dir, available_experiments):
    intervals_path = comparison_dir / "paired_bootstrap_intervals.csv"
    if not intervals_path.is_file():
        return None

    intervals = []
    with intervals_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "macro_f1":
                continue
            before = row["before"].zfill(2)
            after = row["after"].zfill(2)
            if before not in available_experiments or after not in available_experiments:
                continue
            values = tuple(float(row[key]) for key in ("delta_pp", "lower_95_pp", "upper_95_pp"))
            if not all(isfinite(value) for value in values) or values[1] > values[2]:
                raise ValueError(f"Invalid confidence interval for experiment pair {before} → {after}.")
            intervals.append((f"{before} → {after}", *values))

    if not intervals:
        return None

    import matplotlib.pyplot as plt

    labels, deltas, lower, upper = zip(*intervals)
    positions = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 1.6 + 0.42 * len(labels))), layout="constrained")
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.hlines(positions, lower, upper, color="#155e75", linewidth=2)
    ax.scatter(deltas, positions, color="#155e75", s=28, zorder=3)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set(
        title="Paired held-out macro F1 differences",
        xlabel="After minus before (percentage points) with 95% bootstrap intervals",
        ylabel="Experiment pair",
    )
    path = comparison_dir / "paired_macro_f1.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_comparisons(results_root: Path):
    """Write summary CSVs and an optional CI plot, returning the generated paths."""
    comparison_dir = results_root / "Model Comparison Results"
    summaries = {}
    for experiment, folder in discover_experiments(results_root):
        source = folder / "test_results.json"
        if not source.is_file():
            continue
        summary = json.loads(source.read_text(encoding="utf-8"))
        row = {key: summary.get(key) for key in OVERALL_FIELDS}
        row["experiment"] = experiment
        row["architecture"] = folder.name.rpartition(" - ")[0]
        row["training_recipe"] = first_recorded(summary, "training_recipe") or RECIPES.get(experiment)
        row["result_folder"] = folder.name
        row["selected_epoch"] = first_recorded(summary, "selected_epoch", "best_epoch", "best_stage2_epoch")
        if row["selected_stage"] is None and summary.get("best_stage2_epoch") is not None:
            row["selected_stage"] = "stage2"
        row["best_validation_macro_f1"] = first_recorded(
            summary, "selected_validation_macro_f1", "best_validation_macro_f1", "best_stage2_validation_macro_f1"
        )
        row["test_samples"] = first_recorded(summary, "test_samples", "total_test_samples")
        if row["training_minutes"] is None and finite_score(summary.get("training_and_validation_seconds")):
            row["training_minutes"] = summary["training_and_validation_seconds"] / 60
        if isinstance(row["training_hardware"], list):
            row["training_hardware"] = ", ".join(row["training_hardware"])
        summaries[experiment] = row

    comparison_dir.mkdir(exist_ok=True)
    overall_path = comparison_dir / "overall_comparison.csv"
    write_table(overall_path, OVERALL_FIELDS, summaries.values())
    written = [overall_path]

    fields = ("comparison", "before", "after", "metric", "before_score", "after_score", "delta_pp")
    for filename, pairs in COMPARISONS.items():
        rows = []
        for comparison, before, after in pairs:
            if before not in summaries or after not in summaries:
                continue
            for metric in SCORE_FIELDS:
                old = summaries[before][metric]
                new = summaries[after][metric]
                if not finite_score(old) or not finite_score(new):
                    continue
                rows.append({
                    "comparison": comparison,
                    "before": before,
                    "after": after,
                    "metric": metric,
                    "before_score": old,
                    "after_score": new,
                    "delta_pp": 100 * (new - old),
                })
        path = comparison_dir / filename
        write_table(path, fields, rows)
        written.append(path)

    intervals_plot_path = plot_paired_macro_f1(comparison_dir, summaries)
    if intervals_plot_path is not None:
        written.append(intervals_plot_path)
    return written
