import pytest


def test_logistic_regression_classification(client):
    response = client.post("/logistic-regression-classification")
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
    response = client.post("/knn-classification", json={"n_neighbors": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["n_neighbors"] == 3
    assert 0 <= body["f1_score"] <= 1
