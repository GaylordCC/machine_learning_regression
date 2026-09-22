from unittest.mock import patch

import numpy as np
import pytest

from machine_learning.core.exceptions import InvalidTrainingDataError
from machine_learning.schemas import KnnClassificationSchema
from machine_learning.services.classification.knn_service import KnnService

CV_CLASSIFICATION_KEYS = ("cv_f1_mean", "cv_f1_std", "cv_roc_auc_mean", "cv_roc_auc_std")


def _assert_cv_classification_keys(body: dict) -> None:
    for key in CV_CLASSIFICATION_KEYS:
        assert 0 <= body[key] <= 1, key


def test_knn_raises_invalid_training_data_error_when_n_neighbors_exceeds_available_samples():
    """n_neighbors is schema-bounded to 1-50 (well under this dataset's real size),
    so this can't happen through the live endpoint -- but sklearn raises this
    ValueError for real whenever it does, so the mapping needs its own test."""
    tiny_split = (
        np.array([[0, 0], [1, 1]]),  # X_train: 2 samples
        np.array([[0.5, 0.5]]),      # X_test
        np.array([0, 1]),            # Y_train
        np.array([1]),               # Y_test
    )
    with patch(
        "machine_learning.services.classification.knn_service.split_train_test",
        return_value=tiny_split,
    ):
        with pytest.raises(InvalidTrainingDataError):
            KnnService().handle_knn_classification(KnnClassificationSchema(n_neighbors=5))


def test_logistic_regression_classification(client):
    response = client.post("/v1/logistic-regression-classification")
    assert response.status_code == 200
    body = response.json()

    # Fully deterministic: fixed random_state=0 in the train/test split and the
    # StandardScaler fit, plus LogisticRegression's default 'lbfgs' solver (which
    # doesn't depend on random_state). Exact values catch a real regression
    # (e.g. fitting the scaler on test data) instead of just checking the value
    # is "a valid ratio".
    assert body["confusion_matrix"] == [[56, 2], [5, 17]]
    assert body["precision"] == pytest.approx(0.8947368421052632)
    assert body["recall"] == pytest.approx(0.7727272727272727)
    assert body["f1_score"] == pytest.approx(0.8292682926829268)
    assert body["roc_auc"] == pytest.approx(0.9788, abs=1e-3)
    assert 0 <= body["f1_train"] <= 1
    _assert_cv_classification_keys(body)


def test_knn_classification_is_reachable(client):
    """KNN answers on its own route, with no path collision with logistic regression."""
    response = client.post("/v1/knn-classification", json={"n_neighbors": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["n_neighbors"] == 3
    assert 0 <= body["f1_score"] <= 1
    assert 0 <= body["roc_auc"] <= 1
    _assert_cv_classification_keys(body)


def test_knn_with_one_neighbor_memorizes_the_train_set(client):
    """k=1 scores a perfect F1 on train (each point is its own neighbor) but
    less on test: the gap that a test-only metric would not show."""
    body = client.post("/v1/knn-classification", json={"n_neighbors": 1}).json()
    assert body["f1_train"] == pytest.approx(1.0)
    assert body["f1_train"] > body["f1_score"]
