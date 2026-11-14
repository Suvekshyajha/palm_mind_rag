from tests.conftest import Env

TEXT = (
    "Our company gives every employee twenty days of paid leave each year. "
    "Leave must be requested two weeks ahead.\n\n"
    "The office opens at nine and closes at five. Remote work is allowed on Fridays. "
) * 6


def _upload(env: Env, name: str, data: bytes, strategy: str = "recursive") -> "object":
    return env.client.post(
        "/documents/upload",
        files={"file": (name, data)},
        data={"chunking_strategy": strategy},
    )


def test_recursive_upload_returns_201_and_stores_chunks(env: Env) -> None:
    response = _upload(env, "policy.txt", TEXT.encode())
    assert response.status_code == 201  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert body["chunking_strategy"] == "recursive"
    assert body["chunk_count"] == len(env.vector_store.stored) > 0
    assert len(env.session.saved) == 1  # metadata row saved


def test_semantic_upload_works(env: Env) -> None:
    response = _upload(env, "policy.txt", TEXT.encode(), "semantic")
    assert response.status_code == 201  # type: ignore[attr-defined]
    assert response.json()["chunking_strategy"] == "semantic"  # type: ignore[attr-defined]


def test_uploaded_chunks_become_searchable_by_bm25(env: Env) -> None:
    _upload(env, "policy.txt", TEXT.encode())
    assert env.bm25.search("paid leave", limit=1)


def test_unsupported_file_type_is_rejected(env: Env) -> None:
    assert _upload(env, "photo.png", b"data").status_code == 415  # type: ignore[attr-defined]


def test_empty_file_is_rejected(env: Env) -> None:
    assert _upload(env, "empty.txt", b"   ").status_code == 400  # type: ignore[attr-defined]


def test_failed_upload_is_rolled_back(env: Env) -> None:
    env.vector_store.fail_on_add = True
    try:
        _upload(env, "policy.txt", TEXT.encode())
    except RuntimeError:
        pass
    assert env.vector_store.deleted  # chunks removed from Qdrant
    assert not list(env.upload_dir.iterdir())  # file removed from disk
    assert env.session.saved == []