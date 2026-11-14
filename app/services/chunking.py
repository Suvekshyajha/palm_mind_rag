import re
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.schemas import ChunkingStrategy
from app.services.embedder import Embedder
from app.services.extraction import Page

# Recursive chunking tries the biggest separator first, then smaller ones.
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class TextChunk:
    text: str
    index: int
    page: int | None


# ---------------- Recursive chunking ----------------
def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Break text into small pieces (each <= size), keeping the separators."""
    if len(text) <= size:
        return [text]
    if not separators:  # nothing left to split on, so cut hard
        return [text[i : i + size] for i in range(0, len(text), size)]
    separator, rest = separators[0], separators[1:]
    parts = text.split(separator)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += separator
        pieces.extend(_split(part, size, rest) if len(part) > size else [part])
    return pieces


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split on natural breaks (paragraphs, lines, sentences), then merge up to `size`."""
    chunks: list[str] = []
    current = ""
    for piece in _split(text, size, SEPARATORS):
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            # Start the next chunk with the end of this one, cut at a word edge.
            tail = current[-overlap:] if overlap else ""
            current = tail[tail.find(" ") + 1 :] if " " in tail else tail
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


# ---------------- Semantic chunking ----------------
def split_sentences(text: str, max_chars: int) -> list[str]:
    """Break text into sentences. A sentence longer than max_chars is cut hard."""
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            sentences.append(part)
        else:
            sentences.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
    return sentences


def group_by_similarity(
    sentences: list[str],
    vectors: list[list[float]],
    max_chars: int,
    min_chars: int,
    percentile: int,
) -> list[str]:
    """Group neighbouring sentences. Start a new chunk where the topic changes."""
    if len(sentences) < 2:
        return list(sentences)

    arr = np.asarray(vectors)
    # Vectors are normalised, so a dot product is the cosine similarity
    # between each sentence and the next one.
    similarities = (arr[:-1] * arr[1:]).sum(axis=1)
    # The weakest links (lowest `percentile` percent) are the topic changes.
    cutoff = float(np.percentile(similarities, percentile))

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for i, sentence in enumerate(sentences):
        if current:
            too_long = size + 1 + len(sentence) > max_chars
            topic_changed = similarities[i - 1] < cutoff and size >= min_chars
            if too_long or topic_changed:
                chunks.append(" ".join(current))
                current, size = [], 0
        size += len(sentence) + (1 if current else 0)
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


async def semantic_chunks(text: str, embedder: Embedder, settings: Settings) -> list[str]:
    sentences = split_sentences(text, settings.semantic_max_chars)
    if len(sentences) < 2:
        return sentences
    vectors = await embedder.embed_documents(sentences)
    return group_by_similarity(
        sentences,
        vectors,
        settings.semantic_max_chars,
        settings.semantic_min_chars,
        settings.semantic_break_percentile,
    )


# ---------------- Entry point ----------------
async def chunk_pages(
    pages: list[Page], strategy: ChunkingStrategy, settings: Settings, embedder: Embedder
) -> list[TextChunk]:
    """Chunk each page on its own so every chunk keeps its page number."""
    chunks: list[TextChunk] = []
    for page in pages:
        if strategy == ChunkingStrategy.SEMANTIC:
            texts = await semantic_chunks(page.text, embedder, settings)
        else:
            texts = recursive_chunks(page.text, settings.recursive_chunk_size, settings.recursive_chunk_overlap)
        for text in texts:
            chunks.append(TextChunk(text=text, index=len(chunks), page=page.number))
    return chunks