"""Compare completed PyTorch and Keras validation experiments.

Run after both training scripts have been executed with the same --epochs.
Does not retrain models or use held-out test results to select a winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = ["Run", "Hidden Layers", "Learning Rate", "Best Epoch", "Validation Loss", "Validation MAE"]


def read_experiments(path: Path, framework: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {framework} results: {path}. Run its training script first.")
    frame = pd.read_csv(path)
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"{framework} results missing columns: {sorted(missing)}")
    if frame["Run"].duplicated().any():
        raise ValueError(f"Duplicate experiment names in {path}")
    return frame[REQUIRED].copy()


def compare_reports(reports: Path) -> pd.DataFrame:
    torch_path = reports / "experiments.csv"
    keras_path = reports / "keras" / "experiments.csv"
    torch_df = read_experiments(torch_path, "PyTorch")
    keras_df = read_experiments(keras_path, "Keras")

    torch_metrics = reports / "metrics.json"
    keras_metrics = reports / "keras" / "metrics.json"
    if not torch_metrics.is_file() or not keras_metrics.is_file():
        raise FileNotFoundError("Missing metrics.json; run both training scripts first.")
    torch_meta = json.loads(torch_metrics.read_text(encoding="utf-8"))
    keras_meta = json.loads(keras_metrics.read_text(encoding="utf-8"))
    for key in ("seed", "epochs_per_run"):
        if torch_meta.get(key) != keras_meta.get(key):
            raise ValueError(
                f"Incompatible {key}: PyTorch={torch_meta.get(key)!r}, "
                f"Keras={keras_meta.get(key)!r}. Re-run both with matching settings."
            )

    combined = torch_df.merge(
        keras_df,
        on="Run",
        suffixes=("_PyTorch", "_Keras"),
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not (combined["_merge"] == "both").all():
        raise ValueError("The frameworks have different experiment names.")
    for column in ("Hidden Layers", "Learning Rate"):
        left, right = f"{column}_PyTorch", f"{column}_Keras"
        if column == "Learning Rate":
            matches = (combined[left] - combined[right]).abs() < 1e-12
        else:
            matches = combined[left] == combined[right]
        if not matches.all():
            raise ValueError(f"Experiment settings differ for {column}.")

    result = pd.DataFrame({
        "Run": combined["Run"],
        "Hidden Layers": combined["Hidden Layers_PyTorch"],
        "Learning Rate": combined["Learning Rate_PyTorch"],
        "PyTorch Best Epoch": combined["Best Epoch_PyTorch"],
        "Keras Best Epoch": combined["Best Epoch_Keras"],
        "PyTorch Validation MSE": combined["Validation Loss_PyTorch"],
        "Keras Validation MSE": combined["Validation Loss_Keras"],
        "PyTorch Validation MAE": combined["Validation MAE_PyTorch"],
        "Keras Validation MAE": combined["Validation MAE_Keras"],
    })
    result["MSE Difference (Keras - PyTorch)"] = (
        result["Keras Validation MSE"] - result["PyTorch Validation MSE"]
    )
    result["MAE Difference (Keras - PyTorch)"] = (
        result["Keras Validation MAE"] - result["PyTorch Validation MAE"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = parser.parse_args()
    comparison = compare_reports(args.reports)
    output = args.reports / "framework_comparison.csv"
    comparison.to_csv(output, index=False)
    print("\nValidation comparison (same seed and epoch budget):")
    print(comparison.round(4).to_string(index=False))
    print("\nLower MSE/MAE is better. Differences are Keras minus PyTorch.")
    print("Note: training batch order and optimizer numerics may differ between frameworks.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
