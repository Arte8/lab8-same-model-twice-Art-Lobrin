"""Keras equivalent of the PyTorch bike-rental network."""

import keras
import torch


def build_keras_model(
    input_size: int,
    hidden1: int = 64,
    hidden2: int = 32,
) -> keras.Model:
    """Build a two-hidden-layer regression network."""
    inputs = keras.Input(shape=(input_size,))

    x = keras.layers.Dense(hidden1, activation="relu")(inputs)
    x = keras.layers.Dense(hidden2, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def copy_torch_weights_to_keras(
    torch_model: torch.nn.Module,
    keras_model: keras.Model,
) -> keras.Model:
    """Copy PyTorch Linear weights and biases into Keras Dense layers."""

    torch_layers = [layer for layer in torch_model.modules() if isinstance(layer, torch.nn.Linear)]

    keras_layers = [layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)]

    if len(torch_layers) != len(keras_layers):
        raise ValueError("Different number of Linear/Dense layers.")

    for torch_layer, keras_layer in zip(torch_layers, keras_layers):
        if torch_layer.bias is None:
            raise ValueError("PyTorch Linear layer has no bias.")

        weights = torch_layer.weight.detach().cpu().numpy().T
        bias = torch_layer.bias.detach().cpu().numpy()

        keras_layer.set_weights([weights, bias])

    return keras_model
