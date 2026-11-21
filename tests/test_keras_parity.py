"""Lab 08: PyTorch/Keras architecture, weight-copy, and training smoke tests."""

import numpy as np
import pytest
import torch

from keras_model import build_keras_model, copy_torch_weights_to_keras
from model import BikeRentalNN
from train_keras import measure, train_one


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
