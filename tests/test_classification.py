from unittest.mock import patch

import numpy as np
import pytest

from machine_learning.core.exceptions import InvalidTrainingDataError
from machine_learning.schemas import KnnClassificationSchema
from machine_learning.services.classification.knn_service import KnnService


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
        "machine_learning.services.classification.knn_service.prepare_train_test_split",
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
    # (e.g. reintroducing the fit_transform-on-test data leakage bug) instead of
    # just checking the value is "a valid ratio".
    assert body["confusion_matrix"] == [[56, 2], [5, 17]]
    assert body["precision"] == pytest.approx(0.8947368421052632)
    assert body["recall"] == pytest.approx(0.7727272727272727)
    assert body["f1_score"] == pytest.approx(0.8292682926829268)


def test_knn_classification_is_reachable(client):
    """Regression test for the duplicate-route bug that made this endpoint unreachable."""
    response = client.post("/v1/knn-classification", json={"n_neighbors": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["n_neighbors"] == 3
    assert 0 <= body["f1_score"] <= 1
