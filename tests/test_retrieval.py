from app.services.retrieval import reciprocal_rank_fusion


def test_chunk_found_by_both_searches_ranks_first() -> None:
    a, b, c = {"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}
    fused = reciprocal_rank_fusion([[a, b], [b, c]], k=60)
    assert fused[0]["chunk_id"] == "b"
    assert len(fused) == 3