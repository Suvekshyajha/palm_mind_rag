import asyncio
import contextlib
import time
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import Document, save_document
from app.schemas import ChunkingStrategy, IngestionResponse
from app.services.chunking import chunk_pages
from app.services.cleaning import clean_text
from app.services.embedder import Embedder
from app.services.extraction import Page, extract_pages
from app.services.retrieval import BM25Index
from app.services.vector_store import VectorStore
from app.typedefs import ChunkPayload

ALLOWED_TYPES = {".pdf": "pdf", ".txt": "txt", ".docx": "docx"}


class IngestionError(Exception):
    """A problem with the uploaded file. Carries the HTTP status to return."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


async def _validate_and_read(file: UploadFile, settings: Settings) -> tuple[str, str, bytes]:
    """Check the file and return (filename, file_type, content)."""
    filename = Path(file.filename or "").name  # drop any folder parts
    extension = Path(filename).suffix.lower()
    file_type = ALLOWED_TYPES.get(extension)
    if file_type is None:
        raise IngestionError("Only .pdf, .txt and .docx files are supported.", 415)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)  # read one extra byte to detect too-big files
    if len(data) > max_bytes:
        raise IngestionError(f"File is larger than {settings.max_upload_mb} MB.", 413)
    if not data.strip():
        raise IngestionError("File is empty.")
    if file_type == "pdf" and not data.startswith(b"%PDF"):
        raise IngestionError("This file is not a valid PDF.")
    return filename, file_type, data


async def ingest_document(
    *,
    file: UploadFile,
    strategy: ChunkingStrategy,
    session: AsyncSession,
    embedder: Embedder,
    vector_store: VectorStore,
    bm25: BM25Index,
    settings: Settings,
) -> IngestionResponse:
    """Full upload flow: validate, extract, clean, chunk, embed, store."""
    started = time.perf_counter()
    filename, file_type, data = await _validate_and_read(file, settings)

    # Keep the original file on disk under a unique name.
    document_id = str(uuid.uuid4())
    file_path = Path(settings.upload_dir) / f"{document_id}.{file_type}"
    await asyncio.to_thread(file_path.write_bytes, data)

    try:
        try:
            pages = await asyncio.to_thread(extract_pages, file_path, file_type, settings.enable_ocr)
        except Exception as error:
            raise IngestionError("Could not read this file. It may be corrupted.", 422) from error

        pages = [Page(page.number, clean_text(page.text)) for page in pages]
        chunks = await chunk_pages(pages, strategy, settings, embedder)
        if not chunks:
            raise IngestionError("No readable text was found in this file.", 422)

        vectors = await embedder.embed_documents([chunk.text for chunk in chunks])
        payloads = [
            ChunkPayload(
                document_id=document_id,
                chunk_id=str(uuid.uuid4()),
                chunk_index=chunk.index,
                source=filename,
                page=chunk.page,
                chunking_strategy=strategy.value,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        await vector_store.add_chunks(payloads, vectors)

        page_count = len(pages) if file_type == "pdf" else None
        await save_document(
            session,
            Document(
                id=document_id,
                filename=filename,
                file_type=file_type,
                file_path=str(file_path),
                chunking_strategy=strategy.value,
                chunk_count=len(chunks),
                page_count=page_count,
            ),
        )

    except Exception:
        # Something failed, so undo everything for this upload.
        file_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            await vector_store.delete_document(document_id)
        raise

    # Make the new chunks searchable by BM25 (in memory, Qdrant is not read again).
    await bm25.add(payloads)

    return IngestionResponse(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        chunking_strategy=strategy,
        page_count=page_count,
        chunk_count=len(chunks),
        processing_time_seconds=round(time.perf_counter() - started, 2),
    )