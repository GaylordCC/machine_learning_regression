"""Tests for services/shared/evaluation.py: cross-validation summaries and
binary-classifier scoring, on small synthetic datasets."""
import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from machine_learning.services.shared.evaluation import (
    cross_validated_classification,
    cross_validated_regression,
    evaluate_binary_classifier,
    leave_one_out_regression,
)


@pytest.fixture
def linear_data():
    x = np.arange(1, 21, dtype=float).reshape(-1, 1)
    return x, 2 * x.ravel() + 1


@pytest.fixture
def separable_classes():
    """Two well-separated blobs: any reasonable classifier should score high."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, size=(60, 2)), rng.normal(2, 1, size=(60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    order = rng.permutation(len(y))
    return X[order], y[order]


def test_cross_validated_regression_recovers_a_perfect_linear_relationship(linear_data):
    X, y = linear_data
    result = cross_validated_regression(LinearRegression(), X, y)
    assert result["cv_r2_mean"] == pytest.approx(1.0)
    assert result["cv_rmse_mean"] == pytest.approx(0.0, abs=1e-8)
    assert set(result) == {"cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std"}


def test_cross_validated_regression_uses_fewer_folds_on_tiny_datasets():
    """8 rows cannot fill 5 folds of 2+ samples (R2 is undefined on a single
    sample), so the fold count shrinks instead of returning NaN."""
    X = np.arange(8, dtype=float).reshape(-1, 1)
    result = cross_validated_regression(LinearRegression(), X, 3 * X.ravel())
    assert all(math.isfinite(value) for value in result.values())


def test_cross_validation_is_deterministic(separable_classes):
    X, y = separable_classes
    model = make_pipeline(StandardScaler(), LogisticRegression())
    assert cross_validated_classification(model, X, y) == cross_validated_classification(model, X, y)


def test_cross_validated_classification_reports_f1_and_roc_auc(separable_classes):
    X, y = separable_classes
    model = make_pipeline(StandardScaler(), LogisticRegression())
    result = cross_validated_classification(model, X, y)
    assert set(result) == {"cv_f1_mean", "cv_f1_std", "cv_roc_auc_mean", "cv_roc_auc_std"}
    assert result["cv_f1_mean"] > 0.9
    assert result["cv_roc_auc_mean"] > 0.9
    assert result["cv_f1_std"] >= 0 and result["cv_roc_auc_std"] >= 0


def test_leave_one_out_regression_scores_out_of_sample_predictions(linear_data):
    X, y = linear_data
    result = leave_one_out_regression(LinearRegression(), X, y)
    assert result["r2"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-8)


def test_evaluate_binary_classifier_returns_every_metric(separable_classes):
    X, y = separable_classes
    model = make_pipeline(StandardScaler(), LogisticRegression())
    result = evaluate_binary_classifier(model, X[:90], y[:90], X[90:], y[90:])

    assert set(result) == {
        "confusion_matrix", "precision", "recall", "f1_score", "f1_train", "roc_auc",
        "cv_f1_mean", "cv_f1_std", "cv_roc_auc_mean", "cv_roc_auc_std",
    }
    assert np.array(result["confusion_matrix"]).sum() == 30
    assert result["roc_auc"] > 0.9
    assert 0 <= result["f1_train"] <= 1
