from fastapi.testclient import TestClient

from contextbench.api import create_app


def test_project_ingest_index_and_retrieve(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "db.sqlite", tmp_path / "vectors"))
    response = client.post("/api/v1/projects", json={"name": "Test"})
    assert response.status_code == 201
    project_id = response.json()["id"]
    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"file": ("raft.md", b"# Raft\nA follower catches up from the leader log.")},
    )
    assert response.status_code == 201
    index = client.post(
        f"/api/v1/projects/{project_id}/indexes",
        json={
            "name": "hash",
            "chunking": {"strategy": "PARAGRAPH"},
            "embedding": {"provider": "hash"},
        },
    )
    assert index.status_code == 201, index.text
    index_id = index.json()["id"]
    retrieved = client.post(
        f"/api/v1/projects/{project_id}/retrieve",
        json={
            "query": "follower catches up",
            "indexConfigurationId": index_id,
            "methods": ["BM25", "VECTOR", "HYBRID"],
        },
    )
    assert retrieved.status_code == 200, retrieved.text
    assert retrieved.json()["context"]


def test_consistent_validation_error_shape(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "db.sqlite", tmp_path / "vectors"))
    response = client.post("/api/v1/projects", json={"description": "missing name"})
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
