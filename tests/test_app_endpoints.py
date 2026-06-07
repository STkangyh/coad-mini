"""Smoke tests for the demo API surface (no video / CLIP needed)."""
import app
from fastapi.testclient import TestClient

client = TestClient(app.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_routes_registered():
    paths = {r.path for r in app.app.routes if hasattr(r, "path")}
    for p in ["/predict", "/predict_rt", "/predict_anomaly", "/enroll",
              "/classes", "/reset_classes", "/forgetting"]:
        assert p in paths, f"missing route {p}"


def test_classes_lists_base():
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert data["base_count"] == 48
    assert data["n_classes"] >= 48
    # 'enrolled' only contains ids >= base_count
    assert all(c["id"] >= data["base_count"] for c in data["enrolled"])


def test_reset_classes_returns_to_base():
    r = client.post("/reset_classes")
    assert r.status_code == 200
    assert r.json()["n_classes"] == 48
    # after reset there should be no enrolled classes
    assert client.get("/classes").json()["enrolled"] == []
