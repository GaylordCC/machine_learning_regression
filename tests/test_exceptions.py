"""Tests for the global exception handlers registered in main.py.

Verifies the HTTP-status mapping for the app's domain exceptions, and --
the specific bug this replaces -- that an unexpected error never leaks its
raw message to the client.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from machine_learning.core.exceptions import InvalidTrainingDataError, UpstreamServiceError
from machine_learning.main import app


def test_invalid_training_data_error_maps_to_422_with_its_message(client):
    with patch(
        "machine_learning.services.classification.knn_service.KnnService.handle_knn_classification",
        side_effect=InvalidTrainingDataError("n_neighbors=5 is not valid for this dataset"),
    ):
        response = client.post("/v1/knn-classification", json={"n_neighbors": 5})
    assert response.status_code == 422
    assert response.json()["detail"] == "n_neighbors=5 is not valid for this dataset"


def test_upstream_service_error_maps_to_503_with_its_message(client):
    with patch(
        "machine_learning.services.classification.image_classification_service."
        "ImageClassificationService.handle_classification_image",
        side_effect=UpstreamServiceError("Could not fetch MNIST from OpenML: timed out"),
    ):
        response = client.post("/v1/classification-algorithm")
    assert response.status_code == 503
    assert response.json()["detail"] == "Could not fetch MNIST from OpenML: timed out"


def test_unexpected_error_maps_to_500_without_leaking_the_exception_message():
    # The bare-Exception handler is wired into Starlette's ServerErrorMiddleware,
    # which always re-raises after responding (so servers/tests can still see the
    # traceback) -- the default `client` fixture would surface that raise as a
    # test failure instead of letting us inspect the response it already sent.
    no_raise_client = TestClient(app, raise_server_exceptions=False)
    with patch(
        "machine_learning.services.regression.linear_regression_service."
        "LinearRegressionService.regression_linear_model",
        side_effect=RuntimeError("some internal bug, path=/etc/passwd"),
    ):
        response = no_raise_client.post("/v1/linear-regression", json={"column_name": "TV"})
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert "/etc/passwd" not in body["detail"]
