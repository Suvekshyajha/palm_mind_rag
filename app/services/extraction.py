from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf
from docx import Document as DocxDocument

if TYPE_CHECKING:
    import easyocr


@dataclass
class Page:
    number: int | None  # None for files without pages (txt, docx)
    text: str


def extract_pages(path: Path, file_type: str, enable_ocr: bool) -> list[Page]:
    """Read a file and return its text, page by page when possible."""
    if file_type == "pdf":
        return _read_pdf(path, enable_ocr)
    if file_type == "docx":
        document = DocxDocument(str(path))
        paragraphs = [p.text for p in document.paragraphs]
        return [Page(None, "\n".join(paragraphs))]
    return [Page(None, path.read_text(encoding="utf-8", errors="replace"))]


def _read_pdf(path: Path, enable_ocr: bool) -> list[Page]:
    pages: list[Page] = []
    with pymupdf.open(path) as pdf:  # type: ignore[no-untyped-call, unused-ignore]
        for number, page in enumerate(pdf, start=1):
            text = page.get_text()
            # A page with no text is probably a scan, so try OCR if it is turned on.
            if not text.strip() and enable_ocr:
                text = _ocr_page(page)
            pages.append(Page(number, text))
    return pages


@lru_cache(maxsize=1)
def _get_ocr_reader() -> "easyocr.Reader":  # loaded only when OCR is actually needed
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


def _ocr_page(page: pymupdf.Page) -> str:
    import numpy as np

    pixmap = page.get_pixmap(dpi=200)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    return "\n".join(_get_ocr_reader().readtext(image, detail=0))