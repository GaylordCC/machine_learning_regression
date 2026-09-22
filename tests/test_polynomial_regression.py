import math

from machine_learning.services.regression.polynomial_regression_service import PolynomialRegressionService


def test_polynomial_regression_default_degree(client):
    response = client.post("/v1/polynomial-regression", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["degree"] == 4
    assert 0 <= body["r2_polynomial"] <= 1
    assert body["plot_file"].startswith("polynomicalregression_degree4_")


def test_polynomial_regression_higher_degree_fits_train_better(client):
    """A higher degree should fit this small training set at least as well (overfitting risk)."""
    low = client.post("/v1/polynomial-regression", json={"degree": 1}).json()
    high = client.post("/v1/polynomial-regression", json={"degree": 8}).json()
    assert high["r2_polynomial"] >= low["r2_polynomial"]


def test_leave_one_out_exposes_overfitting_that_the_fit_r2_hides():
    """r2_polynomial (scored on the rows the model trained on) keeps rising with
    the degree, but the leave-one-out R2 collapses: the model memorizes."""
    service = PolynomialRegressionService()
    data = service.build_dataset()
    moderate = service._train(data, degree=4)
    extreme = service._train(data, degree=9)

    assert extreme["r2_polynomial"] >= moderate["r2_polynomial"]
    assert extreme["r2_polynomial_cv"] < moderate["r2_polynomial_cv"]
    assert moderate["r2_polynomial_cv"] > moderate["r2_linear_cv"]


def test_polynomial_regression_endpoint_returns_cross_validated_metrics(client):
    body = client.post("/v1/polynomial-regression", json={"degree": 10}).json()
    for key in ("r2_linear_cv", "rmse_linear_cv", "r2_polynomial_cv", "rmse_polynomial_cv"):
        assert math.isfinite(body[key]), key


def test_polynomial_regression_rejects_invalid_degree(client):
    response = client.post("/v1/polynomial-regression", json={"degree": 0})
    assert response.status_code == 422
