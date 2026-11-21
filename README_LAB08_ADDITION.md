## Lab 08: PyTorch vs. Keras/TensorFlow

This extension reuses the Lab 07 data generator, split, and train-only preprocessing. It adds `keras_model.py` (equivalent architecture and PyTorch-to-Keras weight copying), `train_keras.py` (Keras training and reports), `compare.py` (validation-metric comparison), and the tests in `tests/test_keras_parity.py` and `tests/test_compare.py`.

In Anaconda Prompt from the repository root:

```cmd
conda activate lab8-lbs
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python train.py --epochs 5
python train_keras.py --epochs 5
python compare.py
```

For the full experiment, run both training commands again with `--epochs 100`, then rerun `python compare.py`. Both frameworks must use the same epoch budget; `compare.py` rejects mismatches. **Back up previous reports first** because the scripts overwrite CSV/JSON results.

PyTorch reports are under `reports/`; Keras reports are under `reports/keras/`; the combined comparison is `reports/framework_comparison.csv`. See [COMPARISON.md](COMPARISON.md) for the experiment design, recorded five-epoch results, limitations, and full-run instructions.

The Keras/TensorFlow dependencies must be present in the submitted `environment.yml` and `requirements.txt`. Ensure `lab8-lbs` can be recreated from those files before submission.
