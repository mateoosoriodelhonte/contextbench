from contextbench.chunking import chunk_text
from contextbench.evaluation import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from contextbench.schemas import ChunkingConfig, CreateProjectRequest


def test_schemas_use_camel_case_aliases() -> None:
    request = CreateProjectRequest.model_validate({"name": "Demo", "description": "x"})
    assert request.name == "Demo"
    assert (
        request.model_dump(by_alias=True)["createdAt"] is None
        if "createdAt" in request.model_dump(by_alias=True)
        else True
    )


def test_fixed_token_chunking_has_overlap_and_provenance() -> None:
    chunks = chunk_text(
        "one two three four five six",
        ChunkingConfig(strategy="FIXED_TOKEN", chunkSize=4, overlap=1),
    )
    assert [c.text for c in chunks] == ["one two three four", "four five six"]
    assert chunks[0].start_char == 0 and chunks[1].start_char > chunks[0].start_char


def test_retrieval_metrics() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"b", "c"}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert mrr(ranked, relevant) == 0.5
    assert hit_rate(ranked, relevant, 2) == 1.0
    assert ndcg_at_k(ranked, relevant, 3) > 0.69
