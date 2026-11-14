from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.dependencies import BM25Dep, EmbedderDep, SessionDep, SettingsDep, VectorStoreDep
from app.schemas import ChunkingStrategy, IngestionResponse
from app.services.ingestion import IngestionError, ingest_document

router = APIRouter(prefix="/documents", tags=["Documents"])


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