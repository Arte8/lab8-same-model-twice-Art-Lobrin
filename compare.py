"""Compare saved experiment reports or train the main model across three seeds.

Usage:
    python compare.py                 # Existing report-only comparison
    python compare.py --multi-seed --epochs 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REQUIRED = [
    "Run",
    "Hidden Layers",
    "Learning Rate",
    "Best Epoch",
    "Validation Loss",
    "Validation MAE",
]


def read_experiments(path: Path, framework: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {framework} results: {path}. Run its training script first."
        )
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
    result = pd.DataFrame(
        {
            "Run": combined["Run"],
            "Hidden Layers": combined["Hidden Layers_PyTorch"],
            "Learning Rate": combined["Learning Rate_PyTorch"],
            "PyTorch Best Epoch": combined["Best Epoch_PyTorch"],
            "Keras Best Epoch": combined["Best Epoch_Keras"],
            "PyTorch Validation MSE": combined["Validation Loss_PyTorch"],
            "Keras Validation MSE": combined["Validation Loss_Keras"],
            "PyTorch Validation MAE": combined["Validation MAE_PyTorch"],
            "Keras Validation MAE": combined["Validation MAE_Keras"],
        }
    )
    result["MSE Difference (Keras - PyTorch)"] = (
        result["Keras Validation MSE"] - result["PyTorch Validation MSE"]
    )
    result["MAE Difference (Keras - PyTorch)"] = (
        result["Keras Validation MAE"] - result["PyTorch Validation MAE"]
    )
    return result


def train_seed_pair(seed: int, epochs: int, reports: Path) -> list[dict]:
    """Train two models on identical prepared rows; evaluate each test set once."""
    import keras

    from bikes_data import make_rentals
    from keras_model import build_keras_model
    from model import BikeRentalNN
    from prepare import BATCH_SIZE, make_loaders, prepare_data
    from train import EXPERIMENTS, run_epoch, set_seed, train_model
    from train_keras import arrays, measure, seed_everything

    config = next(c for c in EXPERIMENTS if c["name"] == "Main")
    prepared = prepare_data(make_rentals(), seed=seed)
    x_train, y_train = arrays(prepared, "train")
    x_val, y_val = arrays(prepared, "val")
    x_test, y_test = arrays(prepared, "test")

    # Identical starting weights for both frameworks.
    set_seed(seed)
    initial_torch = BikeRentalNN(x_train.shape[1], config["hidden1"], config["hidden2"])
    initial_state = {
        name: value.detach().clone() for name, value in initial_torch.state_dict().items()
    }

    # PyTorch: reuse the original Lab 07 training and best-weight restoration.
    set_seed(seed)
    torch_model = BikeRentalNN(x_train.shape[1], config["hidden1"], config["hidden2"])
    torch_model.load_state_dict(initial_state)
    train_loader, val_loader, test_loader = make_loaders(prepared, seed=seed, batch_size=BATCH_SIZE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=config["lr"])
    started = time.perf_counter()
    torch_history = train_model(
        torch_model, train_loader, val_loader, loss_fn, optimizer, epochs=epochs
    )
    torch_seconds = time.perf_counter() - started
    best_torch_row = torch_history.loc[torch_history["val_loss"].idxmin()]
    # The original train_model() restores the weights at its best validation epoch.
    _, torch_test_mae = run_epoch(torch_model, test_loader, loss_fn)

    # Keras: compile/fit once, with EarlyStopping and best-model checkpoint.
    keras.backend.clear_session()
    seed_everything(seed)
    keras_model = build_keras_model(x_train.shape[1], config["hidden1"], config["hidden2"])
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    checkpoint_dir = reports / "keras" / "checkpoints" / "multiseed"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"main_seed{seed}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=epochs, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
    ]
    started = time.perf_counter()
    fitted = keras_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    keras_seconds = time.perf_counter() - started
    keras_best_index = int(np.argmin(fitted.history["val_loss"]))
    keras_best_epoch = keras_best_index + 1
    # Use saved best checkpoint for consistent final evaluation.
    keras_model = keras.models.load_model(checkpoint_path)
    _, keras_val_mae = measure(keras_model, x_val, y_val)
    _, keras_test_mae = measure(keras_model, x_test, y_test)
    return [
        {
            "framework": "PyTorch",
            "seed": seed,
            "best_epoch": int(best_torch_row["epoch"]),
            "validation_mae": float(best_torch_row["val_mae"]),
            "test_mae": float(torch_test_mae),
            "seconds": float(torch_seconds),
        },
        {
            "framework": "Keras",
            "seed": seed,
            "best_epoch": keras_best_epoch,
            "validation_mae": float(keras_val_mae),
            "test_mae": float(keras_test_mae),
            "seconds": float(keras_seconds),
        },
    ]


def run_multiseed(reports: Path, seeds: tuple[int, ...], epochs: int) -> pd.DataFrame:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if len(set(seeds)) < 3:
        raise ValueError("At least three distinct seeds are required")
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        print(f"\n=== Seed {seed} | Main architecture | {epochs} epochs ===")
        rows.extend(train_seed_pair(seed, epochs, reports))
    result = pd.DataFrame(rows)
    output = reports / "comparison.csv"
    result.to_csv(output, index=False)
    print("\nThree-seed comparison:\n", result.round(4).to_string(index=False))
    print(f"Saved: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
    )
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    if args.multi_seed:
        run_multiseed(args.reports, tuple(args.seeds), args.epochs)
        return
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
