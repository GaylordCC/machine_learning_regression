"""Smoke tests for the regression endpoints not covered elsewhere.

Each of these trains on a small local CSV (no network I/O), so they stay fast.
Random Forest is called with a small n_estimators to keep the suite quick while
still exercising the real training path.
"""
from machine_learning.services.shared.housing_preprocessing import HOUSING_MODEL_COLUMNS

HOUSING_STEPS = len(HOUSING_MODEL_COLUMNS)


def test_exploratory_analysis_returns_advertising_records(client):
    response = client.post("/machine-learning")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0
    assert "Sales" in body[0]


def test_linear_regression_returns_predictions_and_metrics(client):
    response = client.post("/linear-regression", json={"column_name": "TV"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) > 0
    assert 0 <= body["r2_score"] <= 1
    assert body["rmse"] >= 0


def test_linear_regression_rejects_unknown_column(client):
    response = client.post("/linear-regression", json={"column_name": "not-a-column"})
    assert response.status_code == 422


def test_multi_linear_regression_returns_predictions_and_metrics(client):
    response = client.post("/multi-linear-regression")
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) > 0
    assert 0 <= body["r2_score"] <= 1


def test_svr_regression_returns_predictions_and_metrics(client):
    response = client.post("/svr-regression", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["kernel"] == "rbf"
    assert len(body["predictions"]) > 0
    assert 0 <= body["r2_score"] <= 1


def test_housing_linear_regression_returns_a_score_per_incremental_column_set(client):
    response = client.post("/housing-linear-regression")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "linear_regression"
    assert len(body["scores_by_columns"]) == HOUSING_STEPS


def test_decision_tree_regression_returns_a_score_per_incremental_column_set(client):
    response = client.post("/decision-tree-regression", json={"max_depth": 6})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "decision_tree"
    assert body["max_depth"] == 6
    assert len(body["scores_by_columns"]) == HOUSING_STEPS


def test_random_forest_regression_returns_a_score_per_incremental_column_set(client):
    response = client.post("/random-forest-regression", json={"n_estimators": 5, "max_depth": 6})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "random_forest"
    assert body["n_estimators"] == 5
    assert len(body["scores_by_columns"]) == HOUSING_STEPS
