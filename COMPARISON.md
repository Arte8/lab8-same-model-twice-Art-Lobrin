# Lab 08 — Same Model, Two Frameworks

## PyTorch vs. Keras: Framework Comparison and Recommendation

**Lakeshore Bike Share — Lab 08**

Our team implemented equivalent PyTorch and Keras neural networks to predict hourly bicycle rentals. Both frameworks used the same prepared data, 64/32 hidden-layer architecture, Adam optimizer, batch size of 32, and 100-epoch budget. We compared three random seeds (42, 43, and 44).

**1. Prediction accuracy:** PyTorch achieved an average test mean absolute error (MAE) of 3.0478 rentals, compared with 3.1927 for Keras. PyTorch performed better on seeds 42 and 43, while Keras was slightly better on seed 44. These results indicate comparable predictive accuracy, with a modest advantage for PyTorch in this experiment.

**2. Training speed:** PyTorch required approximately 9.58 seconds per run, compared with 26.03 seconds for Keras. PyTorch was approximately 2.7 times faster in our measured environment. For a weekly prediction job, this reduces execution time and provides additional flexibility for retraining and troubleshooting.

**3. Implementation and debugging:** PyTorch uses an explicit training loop with `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`, making individual training operations accessible for inspection. Keras uses `model.compile()` and `model.fit()`, together with `EarlyStopping` and `ModelCheckpoint`, providing a higher-level training workflow with less manual optimization code.

### Recommendation

I recommend **PyTorch** for Lakeshore Bike Share. Our small team already has a working PyTorch pipeline with automated tests, reproducible preprocessing, and detailed training logs. Its measured training speed is advantageous for a weekly automated job, while its flexibility provides a practical foundation if the team later adopts pretrained models from Hugging Face.

### Trade-offs

Choosing PyTorch means maintaining more explicit code for training, validation, optimization, and checkpoint management. Keras provides convenient built-in training callbacks that may reduce development and maintenance effort for straightforward models.

### Condition That Would Change the Recommendation

I would reconsider Keras if the team decides to retain only a simple tabular prediction model, abandons plans for Hugging Face integration, and finds that reducing maintenance complexity matters more than the measured runtime difference.

Overall, both frameworks meet the prediction objective, but PyTorch offers the stronger combination of measured performance, operational efficiency, and future flexibility for this scenario.

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

## Full 100-Epoch Comparison

Both PyTorch and Keras were trained using the same Lakeshore Bike Share dataset, 60/20/20 data split, random seed (42), preprocessing pipeline, neural network architectures, and epoch budget.

Before training, equivalent weights were copied from PyTorch to Keras. Prediction-equivalence tests passed for all four tested hidden-layer configurations, with maximum numerical differences no greater than approximately 0.00000003.

### Validation Results

| Configuration | PyTorch MSE | Keras MSE | PyTorch MAE | Keras MAE |
|---|---:|---:|---:|---:|
| Main (64/32, LR 0.001) | 23.2745 | 26.6849 | 3.3687 | 3.6874 |
| Lower Learning Rate (64/32, LR 0.0005) | 76.8996 | 69.9846 | 5.4230 | 5.2482 |
| Larger Hidden Layer (128/32, LR 0.001) | 15.3439 | 16.4003 | 2.6079 | 2.6849 |

### Analysis

The larger hidden-layer architecture achieved the lowest validation MSE and MAE in both frameworks.

PyTorch achieved the best overall validation MSE of 15.3439 and MAE of 2.6079, with its best checkpoint occurring at epoch 89. Keras achieved its best validation MSE of 16.4003 and MAE of 2.6849 at epoch 100.

PyTorch performed better in the Main and Larger Hidden Layer experiments, while Keras performed better in the Lower Learning Rate experiment.

Although the networks were numerically equivalent before training, differences in batch ordering, optimizer implementation, and floating-point computation can produce different training trajectories.

### Conclusion

Both PyTorch and Keras successfully implemented the same neural network architecture for bike-rental demand prediction.

PyTorch produced the lowest validation errors in this experiment, but the performance differences were relatively small for the strongest configurations. The results do not establish universal superiority of either framework.

The experiment demonstrates that equivalent architectures can produce comparable, but not necessarily identical, training outcomes across machine learning frameworks.