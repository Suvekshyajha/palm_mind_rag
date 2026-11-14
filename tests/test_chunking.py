from app.services.chunking import group_by_similarity, recursive_chunks, split_sentences


def test_recursive_chunks_prefer_paragraph_breaks() -> None:
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = recursive_chunks(text, size=25, overlap=5)
    assert chunks[0].startswith("First paragraph")
    assert any("Second paragraph" in chunk for chunk in chunks)


def test_recursive_with_zero_overlap_does_not_repeat_text() -> None:
    text = "One. " * 40
    chunks = recursive_chunks(text, size=50, overlap=0)
    assert sum(len(chunk) for chunk in chunks) <= len(text)


def test_empty_text_gives_no_chunks() -> None:
    assert recursive_chunks("", size=100, overlap=10) == []
    assert split_sentences("", max_chars=100) == []


def test_split_sentences_cuts_long_sentence() -> None:
    sentences = split_sentences("a" * 250, max_chars=100)
    assert [len(s) for s in sentences] == [100, 100, 50]


def test_semantic_splits_where_the_topic_changes() -> None:
    sentences = ["Cats purr.", "Cats sleep a lot.", "Stocks fell today.", "Markets are down."]
    # Two sentences about cats, then two about money (made-up vectors).
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    chunks = group_by_similarity(sentences, vectors, max_chars=1000, min_chars=0, percentile=25)
    assert chunks == ["Cats purr. Cats sleep a lot.", "Stocks fell today. Markets are down."]


def test_semantic_respects_max_chars() -> None:
    sentences = ["Same topic here."] * 6
    vectors = [[1.0, 0.0]] * 6
    chunks = group_by_similarity(sentences, vectors, max_chars=40, min_chars=0, percentile=25)
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert len(chunks) > 1