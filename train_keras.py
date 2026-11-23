"""Train Keras bike-rental models with callbacks and reproducible preprocessing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import keras
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bikes_data import make_rentals
from keras_model import build_keras_model
from prepare import BATCH_SIZE, SEED, prepare_data
from train import EXPERIMENTS


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)


def arrays(prepared, split: str):
    x = np.asarray(prepared[f"X_{split}"], dtype=np.float32)
    y = np.asarray(prepared[f"y_{split}"], dtype=np.float32).reshape(-1, 1)
    return x, y


def measure(model, x, y, batch_size=BATCH_SIZE):
    predictions = model.predict(x, batch_size=batch_size, verbose=0)
    errors = predictions.astype(np.float64) - y.astype(np.float64)
    return float(np.mean(errors**2)), float(np.mean(np.abs(errors)))


def train_one(
    config,
    prepared,
    epochs,
    batch_size=BATCH_SIZE,
    seed=SEED,
    checkpoint_dir=None,
    patience=none,

):
    """Fit once with callbacks; return best-weight model, history, best epoch."""

    if patience is None:
        patience = epochs
        
    seed_everything(seed)
    x_train, y_train = arrays(prepared, "train")
    x_val, y_val = arrays(prepared, "val")

    model = build_keras_model(
    x_train.shape[1], config["hidden1"], config["hidden2"]
)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).resolve().parent / "reports" / "keras" / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    slug = config["name"].lower().replace(" ", "_")
    checkpoint = checkpoint_dir / f"{slug}_seed{seed}.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint), monitor="val_loss", save_best_only=True
        ),
    ]
    fit_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(fit_history.history["loss"]) + 1),
            "train_loss": fit_history.history["loss"],
            "val_loss": fit_history.history["val_loss"],
            "val_mae": fit_history.history["val_mae"],
        }
    )
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    print(
        f"{config['name']}: best epoch={best_epoch}, "
        f"val MSE={history['val_loss'].min():.4f}, "
        f"val MAE={history.loc[history['val_loss'].idxmin(), 'val_mae']:.4f}"
    )
    return model, history, best_epoch


def save_learning_curve(history: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Training MSE")
    ax.plot(history["epoch"], history["val_loss"], label="Validation MSE")
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    ax.axvline(best_epoch, linestyle="--", label=f"Best epoch: {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="MSE", title="Keras Learning Curve — Main")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    seed_everything(args.seed)
    prepared = prepare_data(make_rentals(), seed=args.seed)
    reports = Path(__file__).resolve().parent / "reports" / "keras"
    reports.mkdir(parents=True, exist_ok=True)

    best_model = None
    best_config = None
    best_loss = float("inf")
    summary = []
    for config in EXPERIMENTS:
        model, history, best_epoch = train_one(
            config,
            prepared,
            args.epochs,
            seed=args.seed,
            checkpoint_dir=reports / "checkpoints",
        )
        slug = config["name"].lower().replace(" ", "_")
        history.to_csv(reports / f"history_{slug}.csv", index=False)
        if config["name"] == "Main":
            save_learning_curve(history, reports / "learning_curves.png")
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
        "seed": args.seed,
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
