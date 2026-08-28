from fastapi.testclient import TestClient

from machine_learning.main import app

client = TestClient(app)


def test_logistic_regression_classification():
    response = client.post("/logistic-regression-classification")
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["precision"] <= 1
    assert 0 <= body["recall"] <= 1


def test_knn_classification_is_reachable():
    """Regression test for the duplicate-route bug that made this endpoint unreachable."""
    response = client.post("/knn-classification", json={"n_neighbors": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["n_neighbors"] == 3
    assert 0 <= body["f1_score"] <= 1
