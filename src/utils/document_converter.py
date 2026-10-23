"""Convert documents to Markdown with a small on-disk cache."""

import hashlib
import time
from functools import cache
from pathlib import Path
from typing import Protocol

from docling.document_converter import DocumentConverter


class _Document(Protocol):
    def export_to_markdown(self) -> str: ...


class _ConversionResult(Protocol):
    @property
    def document(self) -> _Document: ...


class _DocumentConverter(Protocol):
    def convert(self, *, source: Path) -> _ConversionResult: ...


@cache
def _default_converter() -> DocumentConverter:
    return DocumentConverter()


def _cache_path(cache_dir: Path, document_id: str) -> Path:
    digest = hashlib.sha1(document_id.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.md"


def extract_document(
    document_id: str,
    source_path: str | Path,
    cache_dir: str | Path,
    *,
    converter: _DocumentConverter | None = None,
) -> tuple[str, dict[str, float | bool]]:
    """Convert a local document to Markdown and cache it by document ID."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_path = _cache_path(cache_dir, document_id)

    if text_path.exists():
        text = text_path.read_text(encoding="utf-8")
        return text, {"extract_seconds": 0.0, "extract_cached": True}

    started = time.perf_counter()
    active_converter = converter if converter is not None else _default_converter()
    text = active_converter.convert(source=Path(source_path)).document.export_to_markdown()
    elapsed = time.perf_counter() - started

    if not text.strip():
        raise ValueError(f"No text was extracted from {source_path}")

    text_path.write_text(text, encoding="utf-8")
    return text, {"extract_seconds": elapsed, "extract_cached": False}
