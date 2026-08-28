from fastapi.testclient import TestClient

from machine_learning.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_no_duplicate_routes():
    """Regression test: two endpoints sharing a path used to make KNN unreachable."""
    paths = [route.path for route in app.routes if hasattr(route, "methods")]
    assert len(paths) == len(set(paths)), "Duplicate route path found"
