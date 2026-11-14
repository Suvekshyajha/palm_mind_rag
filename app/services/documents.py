from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Document
from app.services.retrieval import BM25Index
from app.services.vector_store import VectorStore


async def list_documents(session: AsyncSession) -> list[Document]:
    """All uploaded documents, newest first."""
    result = await session.scalars(select(Document).order_by(Document.created_at.desc()))
    return list(result)


async def delete_document(
    document_id: str, session: AsyncSession, vector_store: VectorStore, bm25: BM25Index
) -> bool:
    """Remove a document everywhere. Returns False if it does not exist."""
    document = await session.get(Document, document_id)
    if document is None:
        return False
    await vector_store.delete_document(document_id)
    await bm25.remove(document_id)
    await session.delete(document)
    await session.commit()
    Path(document.file_path).unlink(missing_ok=True)
    return True