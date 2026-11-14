from app.services.retrieval import BM25Index, reciprocal_rank_fusion
from app.typedefs import ChunkPayload


def test_chunk_found_by_both_searches_ranks_first() -> None:
    a, b, c = {"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}
    fused = reciprocal_rank_fusion([[a, b], [b, c]], k=60)  # type: ignore[list-item]
    assert fused[0]["chunk_id"] == "b"
    assert len(fused) == 3


def _chunk(chunk_id: str, text: str) -> ChunkPayload:
    return ChunkPayload(
        document_id="d", chunk_id=chunk_id, chunk_index=0, source="a.txt", page=None,
        chunking_strategy="recursive", text=text,
    )


async def test_bm25_add_makes_new_chunks_searchable_without_qdrant() -> None:
    index = BM25Index()
    assert index.search("anything", 3) == []
    await index.add([_chunk("1", "paid leave is twenty days")])
    await index.add([_chunk("2", "the office opens at nine")])
    await index.add([_chunk("3", "remote work is allowed on fridays")])
    assert index.search("office opens nine", 1)[0]["chunk_id"] == "2"