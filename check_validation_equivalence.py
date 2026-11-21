"""Verify PyTorch and Keras equivalence on validation rows."""

import numpy as np
import torch

from bikes_data import make_rentals
from keras_model import build_keras_model, copy_torch_weights_to_keras
from model import BikeRentalNN
from prepare import SEED, prepare_data


def main():
    torch.manual_seed(SEED)

    # Use the existing preparation pipeline.
    df = make_rentals()
    prepared = prepare_data(df, seed=SEED)

    X_val = np.asarray(prepared["X_val"], dtype=np.float32)
    input_size = X_val.shape[1]

    # Build equivalent networks.
    pytorch_model = BikeRentalNN(
        input_size=input_size,
        hidden1=64,
        hidden2=32,
    )

    keras_model = build_keras_model(
        input_size=input_size,
        hidden1=64,
        hidden2=32,
    )

    # Copy PyTorch weights to Keras.
    copy_torch_weights_to_keras(pytorch_model, keras_model)

    # Predict on the SAME validation rows.
    pytorch_model.eval()

    with torch.no_grad():
        pytorch_predictions = pytorch_model(torch.from_numpy(X_val)).cpu().numpy()

    keras_predictions = keras_model(X_val, training=False).numpy()

    # Confirm predictions match within tolerance.
    np.testing.assert_allclose(
        pytorch_predictions,
        keras_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

    max_difference = np.max(np.abs(pytorch_predictions - keras_predictions))

    print("Validation rows:", len(X_val))
    print("Maximum difference:", f"{max_difference:.10f}")
    print("PASS: PyTorch and Keras predictions are equivalent.")


if __name__ == "__main__":
    main()
