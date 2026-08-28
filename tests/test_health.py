from machine_learning.main import app


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_no_duplicate_routes():
    """Regression test: two endpoints sharing (method, path) used to make KNN unreachable.

    Compares (method, path) pairs, not just path: a resource can legitimately
    expose the same path under different HTTP methods (e.g. GET + POST /health).
    """
    method_path_pairs = [
        (method, route.path)
        for route in app.routes
        if hasattr(route, "methods") and route.methods
        for method in route.methods
    ]
    assert len(method_path_pairs) == len(set(method_path_pairs)), "Duplicate (method, path) route found"
