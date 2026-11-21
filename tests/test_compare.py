"""Lab 08: comparison report validation and metric-difference tests."""

import json

import pandas as pd
import pytest

from compare import compare_reports


def write_reports(root, torch_epochs=5, keras_epochs=5, keras_hidden="64, 32"):
    (root / "keras").mkdir(parents=True)
    columns = {
        "Run": ["Main"],
        "Hidden Layers": ["64, 32"],
        "Learning Rate": [0.001],
        "Best Epoch": [5],
        "Validation Loss": [154.9278],
        "Validation MAE": [7.8898],
    }
    pd.DataFrame(columns).to_csv(root / "experiments.csv", index=False)
    columns["Hidden Layers"] = [keras_hidden]
    columns["Validation Loss"] = [152.5131]
    columns["Validation MAE"] = [8.1216]
    pd.DataFrame(columns).to_csv(root / "keras" / "experiments.csv", index=False)
    (root / "metrics.json").write_text(json.dumps({"seed": 42, "epochs_per_run": torch_epochs}))
    (root / "keras" / "metrics.json").write_text(
        json.dumps({"seed": 42, "epochs_per_run": keras_epochs})
    )


def test_comparison_differences_and_signs(tmp_path):
    write_reports(tmp_path)
    result = compare_reports(tmp_path)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["MSE Difference (Keras - PyTorch)"] == pytest.approx(-2.4147)
    assert row["MAE Difference (Keras - PyTorch)"] == pytest.approx(0.2318)


def test_rejects_mismatched_epoch_budgets(tmp_path):
    write_reports(tmp_path, torch_epochs=100, keras_epochs=5)
    with pytest.raises(ValueError, match="epochs_per_run"):
        compare_reports(tmp_path)


def test_rejects_mismatched_architectures(tmp_path):
    write_reports(tmp_path, keras_hidden="128, 32")
    with pytest.raises(ValueError, match="Hidden Layers"):
        compare_reports(tmp_path)


def test_rejects_missing_reports(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare_reports(tmp_path)
