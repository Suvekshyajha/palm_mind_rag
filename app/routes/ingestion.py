from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.db import Document
from app.dependencies import BM25Dep, EmbedderDep, SessionDep, SettingsDep, VectorStoreDep, require_api_key
from app.errors import BACKEND_ERRORS
from app.schemas import ChunkingStrategy, DocumentInfo, IngestionResponse
from app.services.documents import delete_document, list_documents
from app.services.ingestion import IngestionError, ingest_document

router = APIRouter(prefix="/documents", tags=["Documents"], dependencies=[Depends(require_api_key)])


@router.post("/upload", response_model=IngestionResponse, status_code=201)
async def upload_document(
    session: SessionDep,
    embedder: EmbedderDep,
    vector_store: VectorStoreDep,
    bm25: BM25Dep,
    settings: SettingsDep,
    file: UploadFile = File(..., description="A .pdf, .txt or .docx file"),
    chunking_strategy: ChunkingStrategy = Form(ChunkingStrategy.RECURSIVE),
) -> IngestionResponse:
    """Upload a document, split it into chunks and store it for searching."""
    try:
        return await ingest_document(
            file=file,
            strategy=chunking_strategy,
            session=session,
            embedder=embedder,
            vector_store=vector_store,
            bm25=bm25,
            settings=settings,
        )
    except IngestionError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except BACKEND_ERRORS as error:
        raise HTTPException(status_code=503, detail="A backend service (Qdrant or the database) is unavailable.") from error


@router.get("", response_model=list[DocumentInfo])
async def list_uploaded_documents(session: SessionDep) -> list[Document]:
    """List uploaded documents. Copy a document_id into /chat to search only that file."""
    try:
        return await list_documents(session)
    except BACKEND_ERRORS as error:
        raise HTTPException(status_code=503, detail="The database is unavailable.") from error


@router.delete("/{document_id}", status_code=204)
async def delete_uploaded_document(
    document_id: str, session: SessionDep, vector_store: VectorStoreDep, bm25: BM25Dep
) -> None:
    """Delete a document, its chunks and its stored file."""
    try:
        found = await delete_document(document_id, session, vector_store, bm25)
    except BACKEND_ERRORS as error:
        raise HTTPException(status_code=503, detail="A backend service (Qdrant or the database) is unavailable.") from error
    if not found:
        raise HTTPException(status_code=404, detail="Document not found.")