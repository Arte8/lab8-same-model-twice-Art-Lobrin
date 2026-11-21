"""Check parameter-count equivalence between frameworks."""

from keras_model import build_keras_model
from model import BikeRentalNN


def count_torch_parameters(model):
    return sum(p.numel() for p in model.parameters())


for hidden1, hidden2 in [(16, 8), (32, 16), (64, 32), (128, 32)]:
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

    torch_count = count_torch_parameters(torch_model)
    keras_count = keras_model.count_params()

    assert torch_count == keras_count, f"Mismatch: {torch_count} != {keras_count}"

    print(f"{hidden1}/{hidden2}: PyTorch={torch_count}, Keras={keras_count} MATCH")
