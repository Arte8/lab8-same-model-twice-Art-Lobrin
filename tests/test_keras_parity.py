"""Lab 08: PyTorch/Keras architecture, weight-copy, and training smoke tests."""

import keras
import numpy as np
import pytest
import torch

from bikes_data import make_rentals
from keras_model import build_keras_model, copy_torch_weights_to_keras
from model import BikeRentalNN
from prepare import BikeRentalDataset, prepare_data
from train_keras import arrays, measure, train_one


@pytest.mark.parametrize("hidden1,hidden2", [(16, 8), (32, 16), (64, 32), (128, 32)])
def test_parameter_counts_match(hidden1, hidden2):
    torch_model = BikeRentalNN(input_size=8, hidden1=hidden1, hidden2=hidden2)
    keras_model = build_keras_model(input_size=8, hidden1=hidden1, hidden2=hidden2)
    torch_count = sum(param.numel() for param in torch_model.parameters())
    assert torch_count == keras_model.count_params()
    assert torch_count == (8 + 1) * hidden1 + (hidden1 + 1) * hidden2 + (hidden2 + 1)


@pytest.mark.parametrize("hidden1,hidden2", [(16, 8), (32, 16), (64, 32), (128, 32)])
def test_copied_weights_produce_equivalent_predictions(hidden1, hidden2):
    torch.manual_seed(42)
    torch_model = BikeRentalNN(input_size=8, hidden1=hidden1, hidden2=hidden2)
    keras_model = build_keras_model(input_size=8, hidden1=hidden1, hidden2=hidden2)
    copy_torch_weights_to_keras(torch_model, keras_model)
    features = np.random.default_rng(42).normal(size=(20, 8)).astype(np.float32)
    torch_model.eval()
    with torch.no_grad():
        torch_pred = torch_model(torch.from_numpy(features)).cpu().numpy()
    keras_pred = keras_model(features, training=False).numpy()
    np.testing.assert_allclose(torch_pred, keras_pred, rtol=1e-5, atol=1e-5)


def test_weight_copy_rejects_mismatched_architecture():
    torch_model = BikeRentalNN(input_size=8, hidden1=16, hidden2=8)
    keras_model = build_keras_model(input_size=8, hidden1=16, hidden2=8)
    # Different input feature dimensions cause an incompatible weight shape.
    wrong_torch_model = BikeRentalNN(input_size=7, hidden1=16, hidden2=8)
    with pytest.raises((ValueError, TypeError)):
        copy_torch_weights_to_keras(wrong_torch_model, keras_model)
    assert sum(p.numel() for p in torch_model.parameters()) == keras_model.count_params()


def test_measure_is_mse_and_mae_in_original_units():
    model = build_keras_model(input_size=8, hidden1=4, hidden2=2)
    for layer in model.layers:
        if hasattr(layer, "kernel"):
            layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])
    x = np.zeros((3, 8), dtype=np.float32)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    mse, mae = measure(model, x, y, batch_size=2)
    assert mse == pytest.approx(14 / 3)
    assert mae == pytest.approx(2.0)


def test_one_epoch_keras_training_smoke():
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(16, 8)).astype(np.float32)
    x_val = rng.normal(size=(8, 8)).astype(np.float32)
    prepared = {
        "X_train": x_train,
        "y_train": (x_train[:, 0] * 2 + 3).astype(np.float32),
        "X_val": x_val,
        "y_val": (x_val[:, 0] * 2 + 3).astype(np.float32),
    }
    config = {"name": "Smoke", "hidden1": 16, "hidden2": 8, "lr": 0.001}
    model, history, best_epoch = train_one(config, prepared, epochs=1, batch_size=8)
    assert best_epoch == 1
    assert len(history) == 1
    assert set(["epoch", "train_loss", "val_loss", "val_mae"]).issubset(history.columns)
    assert np.isfinite(history[["train_loss", "val_loss", "val_mae"]].to_numpy()).all()
    assert model is not None


def test_keras_returns_one_value_per_row():
    """Keras must produce exactly one prediction per observation."""
    model = build_keras_model(input_size=8, hidden1=16, hidden2=8)

    x = np.random.default_rng(42).normal(size=(12, 8)).astype(np.float32)
    predictions = model.predict(x, verbose=0)

    assert predictions.shape == (12, 1)
    assert np.isfinite(predictions).all()


def test_keras_save_reload_predictions(tmp_path):
    """Saving and reloading must preserve model predictions."""

    keras.utils.set_random_seed(42)
    model = build_keras_model(input_size=8, hidden1=16, hidden2=8)

    x = np.random.default_rng(42).normal(size=(10, 8)).astype(np.float32)

    original_predictions = model.predict(x, verbose=0)

    model_path = tmp_path / "test_model.keras"
    model.save(model_path)

    assert model_path.exists()

    loaded_model = keras.models.load_model(model_path)
    loaded_predictions = loaded_model.predict(x, verbose=0)

    np.testing.assert_allclose(
        original_predictions,
        loaded_predictions,
        rtol=1e-6,
        atol=1e-6,
    )


def test_weight_copy_transposes_first_kernel():
    """Keras kernel must equal the transpose of the PyTorch weights."""
    torch.manual_seed(42)

    torch_model = BikeRentalNN(input_size=8, hidden1=16, hidden2=8)
    keras_model = build_keras_model(input_size=8, hidden1=16, hidden2=8)

    copy_torch_weights_to_keras(torch_model, keras_model)

    torch_kernel = next(
        layer.weight.detach().cpu().numpy()
        for layer in torch_model.modules()
        if isinstance(layer, torch.nn.Linear)
    )

    keras_kernel = next(
        layer.kernel.numpy() for layer in keras_model.layers if hasattr(layer, "kernel")
    )

    assert torch_kernel.shape == (16, 8)
    assert keras_kernel.shape == (8, 16)

    np.testing.assert_allclose(
        keras_kernel,
        torch_kernel.T,
        rtol=1e-6,
        atol=1e-6,
    )


def test_frameworks_use_identical_rows():
    """Keras arrays and PyTorch tensors must contain identical rows."""
    prepared = prepare_data(make_rentals(), seed=42)

    for split in ("train", "val", "test"):
        keras_x, keras_y = arrays(prepared, split)

        torch_dataset = BikeRentalDataset(
            prepared[f"X_{split}"],
            prepared[f"y_{split}"],
        )

        np.testing.assert_array_equal(
            keras_x,
            torch_dataset.X.numpy(),
        )
        np.testing.assert_array_equal(
            keras_y,
            torch_dataset.y.numpy(),
        )

        assert len(keras_x) == len(torch_dataset)


def test_keras_memorizes_small_batch():
    """Keras must learn a tiny batch to low training error."""
    keras.utils.set_random_seed(42)

    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 8)).astype(np.float32)
    y = (2 * x[:, :1] - 3 * x[:, 1:2] + 1).astype(np.float32)

    model = build_keras_model(
        input_size=8,
        hidden1=32,
        hidden2=16,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
    )

    initial_mse = float(model.evaluate(x, y, verbose=0))

    model.fit(
        x,
        y,
        epochs=300,
        batch_size=4,
        shuffle=False,
        verbose=0,
    )

    final_mse = float(model.evaluate(x, y, verbose=0))

    assert final_mse < initial_mse
    assert final_mse < 0.05


def test_keras_restores_best_validation_weights(tmp_path):
    """Returned model must reproduce the lowest validation MSE."""
    rng = np.random.default_rng(42)

    x_train = rng.normal(size=(24, 8)).astype(np.float32)
    x_val = rng.normal(size=(12, 8)).astype(np.float32)

    prepared = {
        "X_train": x_train,
        "y_train": (2 * x_train[:, 0] + 1).astype(np.float32),
        "X_val": x_val,
        "y_val": (2 * x_val[:, 0] + 1).astype(np.float32),
    }

    config = {
        "name": "Restore Test",
        "hidden1": 16,
        "hidden2": 8,
        "lr": 0.001,
    }

    model, history, best_epoch = train_one(
        config,
        prepared,
        epochs=12,
        batch_size=8,
        seed=42,
        checkpoint_dir=tmp_path,
    )

    x_val_array, y_val_array = arrays(prepared, "val")

    restored_mse, _ = measure(model, x_val_array, y_val_array)
    recorded_best_mse = float(history["val_loss"].min())

    assert best_epoch == int(history.loc[history["val_loss"].idxmin(), "epoch"])

    assert restored_mse == pytest.approx(
        recorded_best_mse,
        rel=1e-4,
        abs=1e-4,
    )

    checkpoint = tmp_path / "restore_test_seed42.keras"
    assert checkpoint.exists()

    saved_model = keras.models.load_model(checkpoint)
    saved_mse, _ = measure(saved_model, x_val_array, y_val_array)

    assert saved_mse == pytest.approx(
        recorded_best_mse,
        rel=1e-4,
        abs=1e-4,
    )


def test_keras_same_seed_reproduces_short_run(tmp_path):
    """Two independent training runs with the same seed must agree."""
    rng = np.random.default_rng(42)

    x_train = rng.normal(size=(24, 8)).astype(np.float32)
    x_val = rng.normal(size=(12, 8)).astype(np.float32)

    prepared = {
        "X_train": x_train,
        "y_train": (x_train[:, 0] * 2 + 3).astype(np.float32),
        "X_val": x_val,
        "y_val": (x_val[:, 0] * 2 + 3).astype(np.float32),
    }

    config = {
        "name": "Seed Test",
        "hidden1": 16,
        "hidden2": 8,
        "lr": 0.001,
    }

    model1, history1, best_epoch1 = train_one(
        config,
        prepared,
        epochs=3,
        batch_size=8,
        seed=42,
        checkpoint_dir=tmp_path / "run1",
    )

    model2, history2, best_epoch2 = train_one(
        config,
        prepared,
        epochs=3,
        batch_size=8,
        seed=42,
        checkpoint_dir=tmp_path / "run2",
    )

    x_val_array, _ = arrays(prepared, "val")

    predictions1 = model1.predict(x_val_array, verbose=0)
    predictions2 = model2.predict(x_val_array, verbose=0)

    assert best_epoch1 == best_epoch2

    np.testing.assert_allclose(
        history1[["train_loss", "val_loss", "val_mae"]].to_numpy(),
        history2[["train_loss", "val_loss", "val_mae"]].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        predictions1,
        predictions2,
        rtol=1e-5,
        atol=1e-5,
    )
