
"""Verify PyTorch and Keras prediction equivalence."""

import numpy as np
import torch

from keras_model import (
    build_keras_model,
    copy_torch_weights_to_keras,
)
from model import BikeRentalNN


torch.manual_seed(42)

for hidden1, hidden2 in [
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 32),
]:
    torch_model = BikeRentalNN(
        input_size=8,
        hidden1=hidden1,
        hidden2=hidden2,
    )

    keras_model = build_keras_model(
        input_size=8,
        hidden1=hidden1,
        hidden2=hidden2,
    )

    copy_torch_weights_to_keras(
        torch_model,
        keras_model,
    )

    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 8)).astype(np.float32)

    torch_model.eval()

    with torch.no_grad():
        torch_predictions = (
            torch_model(torch.from_numpy(X))
            .cpu()
            .numpy()
        )

    keras_predictions = (
        keras_model(X, training=False).numpy()
    )

    np.testing.assert_allclose(
        torch_predictions,
        keras_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

    max_difference = np.max(
        np.abs(torch_predictions - keras_predictions)
    )

    print(
        f"{hidden1}/{hidden2}: "
        f"PASS | Max difference: {max_difference:.8f}"
    )
