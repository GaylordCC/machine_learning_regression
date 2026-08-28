import pytest
from fastapi.testclient import TestClient

from machine_learning.main import app


@pytest.fixture
def client():
    return TestClient(app)
