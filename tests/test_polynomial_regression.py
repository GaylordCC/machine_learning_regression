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


def test_polynomial_regression_rejects_invalid_degree(client):
    response = client.post("/v1/polynomial-regression", json={"degree": 0})
    assert response.status_code == 422
