# Lab 08 — Same Model, Two Frameworks

## Objective

Implement equivalent feed-forward neural networks in PyTorch and Keras/TensorFlow and compare their behavior under the same data preparation, architectures, learning rates, and epoch budgets. A numerical equivalence test establishes that the networks compute the same function **before training**, after PyTorch parameters are copied into Keras.

## Data and controlled conditions

- Data: instructor-provided `make_rentals()` synthetic Lakeshore Bike Share data, 6,000 rows.
- Split: 60% train / 20% validation / 20% held-out test, seed 42.
- Preparation: `StandardScaler` on five numeric features and `OneHotEncoder` on `station_type`; preprocessing fitted on training rows only.
- Model: 8 input features, two ReLU hidden layers, one linear output predicting rentals.
- Configurations: Main (64/32, Adam 0.001); Lower Learning Rate (64/32, Adam 0.0005); Larger Hidden Layer (128/32, Adam 0.001).
- Loss: MSE; validation metrics: MSE and MAE (rental counts); batch size 32.
- Best epoch: minimum validation MSE. Test set is not used to choose configurations.

## Architecture and initial-function equivalence

The PyTorch `Linear` kernel has shape `(output, input)`, while Keras `Dense` expects `(input, output)`. Copying each kernel **transposed** and copying the bias unchanged produces equivalent predictions.

| Hidden layers | Maximum absolute prediction difference | Result |
|---|---:|---|
| 16 / 8 | 0.00000001 | PASS |
| 32 / 16 | 0.00000001 | PASS |
| 64 / 32 | 0.00000001 | PASS |
| 128 / 32 | 0.00000003 | PASS |

These differences are consistent with float32 rounding. This establishes forward-pass equivalence, **not** identical gradient updates or identical trained models.

## Short-run results: five epochs

Values below are from the recorded `compare.py` output. Lower is better.

| Configuration | PyTorch val MSE | Keras val MSE | PyTorch val MAE | Keras val MAE |
|---|---:|---:|---:|---:|
| Main | 154.9278 | 152.5131 | 7.8898 | 8.1216 |
| Lower Learning Rate | 180.0937 | 179.4957 | 8.9815 | 8.9646 |
| Larger Hidden Layer | 130.7209 | 130.0391 | 7.2195 | 7.4100 |

Keras minus PyTorch validation MSE was −2.4147, −0.5980, and −0.6819, respectively (based on unrounded saved values). Keras had lower MSE in all three runs; PyTorch had lower MAE in two. **Both selected the 128/32 configuration**. All configurations had their lowest validation MSE at epoch 5, so the short run does not establish convergence or general superiority.

## Important methodological differences and limitations

1. Both implementations use the same rows and fitted preprocessing, but **PyTorch and Keras use different batch-shuffling mechanisms**. Matching seed 42 does not guarantee the same mini-batch order.
2. Both use Adam, but framework defaults and floating-point operation ordering may differ. Equal starting weights do not imply equal parameter updates.
3. **Training-loss logging differs**: PyTorch averages losses observed during the training epoch, while the Keras script measures training MSE on the full training set *after* the epoch. Direct comparisons of those training-loss columns are therefore not like-for-like; validation metrics are more appropriate.
4. The reported short-run results cover only one seed and five epochs. Differences are descriptive, not statistical evidence of framework superiority.
5. The short-run scripts each evaluate the held-out test set after validation selection. Future experiment decisions must be based on validation results, **not** these test scores.

## Full-run results — to be completed after execution

Run both frameworks for 100 epochs using the commands below, then regenerate `reports/framework_comparison.csv`. **Do not fill this section with five-epoch values or claim the full run has passed until it actually finishes.**

Record the selected configuration, best epoch, validation MSE/MAE, and held-out test MSE/MAE from the final output. Note any change in relative performance and whether either framework appears to have converged.

## Reproduce

From the project root, with dependencies installed:

```cmd
conda activate lab8-lbs
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python train.py --epochs 100
python train_keras.py --epochs 100
python compare.py
```

`train.py` writes PyTorch results to `reports/`; `train_keras.py` writes Keras results to `reports/keras/`; `compare.py` writes `reports/framework_comparison.csv`. These scripts **overwrite their prior reports**; archive five-epoch outputs before the 100-epoch run if you need to retain them.

## Conclusion (five-epoch evidence only)

The Keras implementation matches the PyTorch network's forward function after weight copying, with maximum observed prediction discrepancy 3×10⁻⁸. Their short-run validation performance is close but not identical. Both frameworks favor the larger 128/32 network. A controlled 100-epoch comparison is required before drawing a conclusion about longer-run convergence.
