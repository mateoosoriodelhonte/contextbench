from fastapi.testclient import TestClient
from sqlalchemy import func, select

from contextbench.api import create_app
from contextbench.db import create_session_factory
from contextbench.models import RetrievalRun


def test_project_ingest_index_and_retrieve(tmp_path) -> None:
    db_path = tmp_path / "db.sqlite"
    client = TestClient(create_app(db_path, tmp_path / "vectors"), base_url="http://localhost")
    response = client.post("/api/v1/projects", json={"name": "Test"})
    assert response.status_code == 201
    created_project = response.json()
    assert created_project["documentCount"] == 0
    assert created_project["indexCount"] == 0
    assert created_project["status"] == "READY"
    project_id = created_project["id"]
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
    payload = retrieved.json()
    assert payload["finalContext"]
    assert payload["contextMethod"] == "HYBRID"
    assert [lane["method"] for lane in payload["lanes"]] == ["VECTOR", "BM25", "HYBRID"]
    assert payload["stageLatency"]["totalMs"] >= 0
    assert set(payload["stageLatency"]) == {
        "vectorMs",
        "bm25Ms",
        "fusionMs",
        "rerankMs",
        "assembleMs",
        "totalMs",
    }
    relevant_id = payload["lanes"][-1]["hits"][0]["chunk"]["id"]
    evaluation = client.post(
        f"/api/v1/projects/{project_id}/evaluation-queries",
        json={
            "query": "How does a follower catch up?",
            "relevantChunkIds": [relevant_id],
            "datasetVersion": 1,
            "notes": "Known fixture answer",
        },
    )
    assert evaluation.status_code == 201, evaluation.text
    experiment = client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={
            "name": "hybrid-fixture",
            "datasetVersion": 1,
            "indexConfigurationId": index_id,
            "method": "HYBRID",
            "kValues": [1, 3, 5],
        },
    )
    assert experiment.status_code == 201, experiment.text
    detail = client.get(f"/api/v1/experiments/{experiment.json()['id']}")
    assert detail.json()["metrics"]["recall@1"] == 1.0
    experiment_export = client.get(f"/api/v1/experiments/{experiment.json()['id']}/export").json()
    assert experiment_export["schema"] == "contextbench.experiment.v1"
    assert experiment_export["id"] == experiment.json()["id"]
    assert "experiment" not in experiment_export
    assert experiment_export["executedAt"]
    overview = client.get(f"/api/v1/projects/{project_id}").json()
    assert overview["totalIndexedTokens"] > 0
    assert overview["evaluationQueryCount"] == 1
    assert overview["latestExperiment"]["metrics"]["recall@1"] == 1.0
    exported = client.get(
        f"/api/v1/projects/{project_id}/evaluation-queries/export?dataset_version=1"
    )
    assert exported.json()["schema"] == "contextbench.evaluation.v1"
    assert exported.json()["queries"][0]["notes"] == "Known fixture answer"
    imported = client.post(
        f"/api/v1/projects/{project_id}/evaluation-queries/import",
        json=exported.json(),
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["imported"] == 1
    with create_session_factory(db_path)() as session:
        assert session.scalar(select(func.count()).select_from(RetrievalRun)) == 1


def test_multiple_index_configurations_keep_their_own_chunks(tmp_path) -> None:
    client = TestClient(
        create_app(tmp_path / "db.sqlite", tmp_path / "vectors"),
        base_url="http://localhost",
    )
    project_id = client.post("/api/v1/projects", json={"name": "Frozen indexes"}).json()["id"]
    client.post(
        f"/api/v1/projects/{project_id}/documents",
        data={"tags": "consensus, fixture"},
        files={
            "file": (
                "notes.md",
                b"# Election\nA candidate requests votes from peers.\n\n"
                b"# Repair\nA follower catches up from the leader log.",
            )
        },
    )

    first = client.post(
        f"/api/v1/projects/{project_id}/indexes",
        json={
            "name": "fixed",
            "chunking": {"strategy": "FIXED_TOKEN", "chunkSize": 5, "overlap": 1},
            "embedding": {"provider": "hash"},
        },
    ).json()
    second = client.post(
        f"/api/v1/projects/{project_id}/indexes",
        json={
            "name": "heading",
            "chunking": {"strategy": "HEADING"},
            "embedding": {"provider": "hash"},
        },
    ).json()

    assert first["id"] != second["id"]
    assert first["chunkCount"] != second["chunkCount"]
    for index_id in (first["id"], second["id"]):
        response = client.post(
            f"/api/v1/projects/{project_id}/retrieve",
            json={
                "query": "follower leader log",
                "indexConfigurationId": index_id,
                "methods": ["BM25", "HYBRID"],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["lanes"][0]["hits"]
    filtered = client.post(
        f"/api/v1/projects/{project_id}/retrieve",
        json={
            "query": "follower leader log",
            "indexConfigurationId": second["id"],
            "methods": ["BM25"],
            "filters": {"tags": ["missing"]},
        },
    )
    assert filtered.status_code == 200
    assert filtered.json()["lanes"][0]["hits"] == []

    judged = client.post(
        f"/api/v1/projects/{project_id}/retrieve",
        json={
            "query": "follower catches up leader log",
            "indexConfigurationId": first["id"],
            "methods": ["BM25"],
        },
    ).json()["lanes"][0]["hits"][0]["chunk"]["id"]
    evaluation = client.post(
        f"/api/v1/projects/{project_id}/evaluation-queries",
        json={
            "query": "follower catches up leader log",
            "relevantChunkIds": [judged],
            "datasetVersion": 1,
        },
    )
    assert evaluation.status_code == 201, evaluation.text
    experiment_ids: list[str] = []
    for index in (first, second):
        experiment = client.post(
            f"/api/v1/projects/{project_id}/experiments",
            json={
                "name": f"bm25-{index['name']}",
                "datasetVersion": 1,
                "indexConfigurationId": index["id"],
                "method": "BM25",
                "kValues": [1],
            },
        )
        assert experiment.status_code == 201, experiment.text
        experiment_ids.append(experiment.json()["id"])
        detail = client.get(f"/api/v1/experiments/{experiment.json()['id']}").json()
        assert detail["metrics"]["recall@1"] == 1.0
        assert detail["configuration"]["datasetSnapshot"]["digest"]
        assert detail["configuration"]["indexSnapshot"]["digest"]
        assert detail["results"][0]["rankings"]["latenciesMs"]["total"] >= 0
    compatible = client.post("/api/v1/experiments/compare", json={"experimentIds": experiment_ids})
    assert compatible.status_code == 200, compatible.text

    client.post(
        f"/api/v1/projects/{project_id}/evaluation-queries",
        json={
            "query": "follower catches up leader log",
            "relevantChunkIds": [judged],
            "datasetVersion": 2,
        },
    )
    incompatible_experiment = client.post(
        f"/api/v1/projects/{project_id}/experiments",
        json={
            "name": "different-dataset",
            "datasetVersion": 2,
            "indexConfigurationId": first["id"],
            "method": "BM25",
            "kValues": [1],
        },
    )
    incompatible = client.post(
        "/api/v1/experiments/compare",
        json={"experimentIds": [experiment_ids[0], incompatible_experiment.json()["id"]]},
    )
    assert incompatible.status_code == 422
    assert incompatible.json()["error"]["code"] == "INCOMPATIBLE_EXPERIMENTS"


def test_consistent_validation_error_shape(tmp_path) -> None:
    client = TestClient(
        create_app(tmp_path / "db.sqlite", tmp_path / "vectors"),
        base_url="http://localhost",
    )
    response = client.post("/api/v1/projects", json={"description": "missing name"})
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    project_id = client.post("/api/v1/projects", json={"name": "Tags"}).json()["id"]
    tags = client.post(
        f"/api/v1/projects/{project_id}/documents",
        data={"tags": "x" * 2001},
        files={"file": ("safe.txt", b"safe text")},
    )
    assert tags.status_code == 422


def test_project_conflict_and_pagination_validation(tmp_path) -> None:
    client = TestClient(
        create_app(tmp_path / "db.sqlite", tmp_path / "vectors"),
        base_url="http://localhost",
    )
    assert client.post("/api/v1/projects", json={"name": "Unique"}).status_code == 201
    conflict = client.post("/api/v1/projects", json={"name": "Unique"})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "CONFLICT"
    assert client.get("/api/v1/projects?page_size=0").status_code == 422
    projects = client.get("/api/v1/projects").json()
    assert projects["data"][0]["documentCount"] == 0


def test_generation_uses_deterministic_no_answer_before_ollama(tmp_path) -> None:
    client = TestClient(
        create_app(tmp_path / "db.sqlite", tmp_path / "vectors"),
        base_url="http://localhost",
    )
    response = client.post(
        "/api/v1/generate",
        json={"question": "What is Raft?", "context": "too short", "minimumEvidenceTokens": 5},
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Retrieved evidence appears insufficient to answer this question.",
        "generated": False,
        "reason": "INSUFFICIENT_EVIDENCE",
    }
    overlap = client.post(
        "/api/v1/generate",
        json={
            "question": "How does Raft elect a leader?",
            "context": " ".join(["unrelated"] * 25),
        },
    )
    assert overlap.status_code == 200
    assert overlap.json()["reason"] == "INSUFFICIENT_QUERY_OVERLAP"


def test_api_rejects_untrusted_host_headers(tmp_path) -> None:
    client = TestClient(
        create_app(tmp_path / "db.sqlite", tmp_path / "vectors"),
        base_url="http://localhost",
    )
    response = client.get("/api/v1/projects", headers={"Host": "attacker.example:8000"})
    assert response.status_code == 400


def test_api_can_serve_a_built_single_page_frontend(tmp_path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>ContextBench</main>", encoding="utf-8")
    client = TestClient(
        create_app(tmp_path / "db.sqlite", tmp_path / "vectors", frontend),
        base_url="http://localhost",
    )
    assert client.get("/projects/demo/query").text == "<main>ContextBench</main>"
    missing_api = client.get("/api/v1/missing")
    assert missing_api.status_code == 404
    assert missing_api.json()["error"]["code"] == "NOT_FOUND"
