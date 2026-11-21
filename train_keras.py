"""Train equivalent Keras bike-rental models without overwriting PyTorch reports."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import torch

from bikes_data import make_rentals
from keras_model import build_keras_model, copy_torch_weights_to_keras
from model import BikeRentalNN
from prepare import BATCH_SIZE, SEED, prepare_data
from train import EXPERIMENTS


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)


def arrays(prepared, split):
    x = np.asarray(prepared[f"X_{split}"], dtype=np.float32)
    y = np.asarray(prepared[f"y_{split}"], dtype=np.float32).reshape(-1, 1)
    return x, y


def measure(model, x, y, batch_size=BATCH_SIZE):
    predictions = model.predict(x, batch_size=batch_size, verbose=0)
    errors = predictions.astype(np.float64) - y.astype(np.float64)
    return float(np.mean(errors**2)), float(np.mean(np.abs(errors)))


def train_one(config, prepared, epochs, batch_size=BATCH_SIZE):
    seed_everything()
    x_train, y_train = arrays(prepared, "train")
    x_val, y_val = arrays(prepared, "val")

    torch_model = BikeRentalNN(x_train.shape[1], config["hidden1"], config["hidden2"])
    model = build_keras_model(x_train.shape[1], config["hidden1"], config["hidden2"])
    copy_torch_weights_to_keras(torch_model, model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    rows = []
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.get_weights())
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        # Keras shuffling is seeded, but its order is not identical to PyTorch's.
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=True, verbose=0)
        train_mse, _ = measure(model, x_train, y_train, batch_size)
        val_mse, val_mae = measure(model, x_val, y_val, batch_size)
        rows.append(
            {"epoch": epoch, "train_loss": train_mse, "val_loss": val_mse, "val_mae": val_mae}
        )
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch
            best_weights = copy.deepcopy(model.get_weights())
        print(
            f"{config['name']} epoch {epoch:03d}/{epochs} | "
            f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )
    model.set_weights(best_weights)
    return model, pd.DataFrame(rows), best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    seed_everything()
    prepared = prepare_data(make_rentals(), seed=SEED)
    reports = Path(__file__).resolve().parent / "reports" / "keras"
    reports.mkdir(parents=True, exist_ok=True)

    best_model = None
    best_config = None
    best_loss = float("inf")
    summary = []
    for config in EXPERIMENTS:
        model, history, best_epoch = train_one(config, prepared, args.epochs)
        slug = config["name"].lower().replace(" ", "_")
        history.to_csv(reports / f"history_{slug}.csv", index=False)
        best_row = history.loc[history["val_loss"].idxmin()]
        summary.append(
            {
                "Run": config["name"],
                "Hidden Layers": f"{config['hidden1']}, {config['hidden2']}",
                "Learning Rate": config["lr"],
                "Best Epoch": best_epoch,
                "Validation Loss": float(best_row["val_loss"]),
                "Validation MAE": float(best_row["val_mae"]),
            }
        )
        if float(best_row["val_loss"]) < best_loss:
            best_loss = float(best_row["val_loss"])
            best_model = model
            best_config = config

    pd.DataFrame(summary).to_csv(reports / "experiments.csv", index=False)
    assert best_model is not None and best_config is not None
    x_test, y_test = arrays(prepared, "test")
    test_mse, test_mae = measure(best_model, x_test, y_test)
    metrics = {
        "seed": SEED,
        "epochs_per_run": args.epochs,
        "selected_model": best_config["name"],
        "selected_hidden_layers": [best_config["hidden1"], best_config["hidden2"]],
        "selected_learning_rate": best_config["lr"],
        "selected_validation_mse": best_loss,
        "test": {"neural_network": {"mse": test_mse, "mae": test_mae}},
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(f"Keras held-out test: MSE={test_mse:.4f}, MAE={test_mae:.4f}")
    print(f"Saved Keras reports in {reports}")


if __name__ == "__main__":
    main()
