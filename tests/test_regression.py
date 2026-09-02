"""Smoke tests for the regression endpoints not covered elsewhere.

Each of these trains on a small local CSV (no network I/O), so they stay fast.
Random Forest is called with a small n_estimators to keep the suite quick while
still exercising the real training path.
"""
import re
import warnings

import pandas as pd

from machine_learning.core.paths import RESULTS_GRAPHICS_DIR
from machine_learning.schemas import RandomForestRegressionSchema
from machine_learning.services.regression.linear_regression_service import LinearRegressionService
from machine_learning.services.regression.svr_service import SvrRegressionService
from machine_learning.services.shared.housing_preprocessing import HOUSING_MODEL_COLUMNS

HOUSING_STEPS = len(HOUSING_MODEL_COLUMNS)


def _assert_plot_file(filename: str, expected_prefix: str) -> None:
    """A plot_file value must point at a real, uniquely-named file on disk,
    with the request's own parameter baked into the name (not just the
    technique) so two different requests never produce the same filename."""
    assert re.match(rf"^{re.escape(expected_prefix)}_\d{{8}}_[0-9a-f]{{8}}\.png$", filename), filename
    assert (RESULTS_GRAPHICS_DIR / filename).exists()


def test_linear_regression_train_simple_is_pure_and_needs_no_disk_io():
    """The training step no longer touches the filesystem or matplotlib -- this
    builds an in-memory DataFrame and never calls load_advertising_data() or
    saved_figure(), unlike the full regression_linear_model() endpoint method."""
    data = pd.DataFrame({
        "TV": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Sales": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    result = LinearRegressionService()._train_simple(data, "TV")
    assert result["r2_score"] > 0.9
    assert len(result["predictions"]) == 2  # 20% of 10 rows held out for test


def test_svr_train_is_pure_and_needs_no_disk_io():
    data = pd.DataFrame({
        "TV": list(range(10, 210, 10)),
        "Radio": list(range(1, 21)),
        "Newspaper": list(range(1, 21)),
        "Sales": list(range(1, 21)),
    })
    result = SvrRegressionService()._train(data, "rbf")
    assert result["kernel"] == "rbf"
    assert len(result["predictions"]) == 4  # 20% of 20 rows held out for test


def test_random_forest_default_n_estimators_is_kept_low_for_latency():
    """Regression test: n_estimators=100 (the old default) took ~45-50s per call
    (measured: 50 estimators = 23.3s) — a bad default for a synchronous HTTP
    endpoint with no progress feedback. Keep it low unless deliberately raised."""
    assert RandomForestRegressionSchema().n_estimators <= 50


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
    _assert_plot_file(body["plot_file"], "plotregression_TV")


def test_linear_regression_plot_file_differs_per_column(client):
    """Two different columns must not collide on the same plot filename --
    the old fixed "plotregression.png" name did exactly this."""
    tv_body = client.post("/linear-regression", json={"column_name": "TV"}).json()
    radio_body = client.post("/linear-regression", json={"column_name": "Radio"}).json()
    assert tv_body["plot_file"] != radio_body["plot_file"]
    assert "TV" in tv_body["plot_file"]
    assert "Radio" in radio_body["plot_file"]


def test_linear_regression_does_not_use_deprecated_squared_param(client):
    """Regression test: mean_squared_error(..., squared=False) is deprecated in
    sklearn 1.4 and removed in 1.6 — root_mean_squared_error replaces it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = client.post("/linear-regression", json={"column_name": "TV"})
    assert response.status_code == 200
    assert not any("squared" in str(w.message) for w in caught)


def test_linear_regression_rejects_unknown_column(client):
    response = client.post("/linear-regression", json={"column_name": "not-a-column"})
    assert response.status_code == 422


def test_multi_linear_regression_returns_predictions_and_metrics(client):
    response = client.post("/multi-linear-regression")
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) > 0
    assert 0 <= body["r2_score"] <= 1
    _assert_plot_file(body["plot_file"], "plotmultiregression")


def test_svr_regression_returns_predictions_and_metrics(client):
    response = client.post("/svr-regression", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["kernel"] == "rbf"
    assert len(body["predictions"]) > 0
    assert 0 <= body["r2_score"] <= 1


def test_svr_regression_does_not_use_deprecated_squared_param(client):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = client.post("/svr-regression", json={})
    assert response.status_code == 200
    assert not any("squared" in str(w.message) for w in caught)


def test_housing_linear_regression_returns_a_score_per_incremental_column_set(client):
    response = client.post("/housing-linear-regression")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "linear_regression"
    assert len(body["scores_by_columns"]) == HOUSING_STEPS
    _assert_plot_file(body["plot_files"]["histograms"], "histograms")
    _assert_plot_file(body["plot_files"]["scatter_plot"], "scatter_plot")
    _assert_plot_file(body["plot_files"]["correlation_plot"], "correlation_plot")


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
