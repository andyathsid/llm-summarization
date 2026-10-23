from pathlib import Path

import pytest

from utils import extract_document


class _FakeDocument:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class _FakeResult:
    def __init__(self, markdown: str) -> None:
        self.document = _FakeDocument(markdown)


class _FakeConverter:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown
        self.sources: list[Path] = []

    def convert(self, *, source: Path) -> _FakeResult:
        self.sources.append(source)
        return _FakeResult(self._markdown)


def test_converts_and_then_uses_cached_markdown(tmp_path: Path) -> None:
    converter = _FakeConverter("# Extracted document")

    text, first_metrics = extract_document(
        "example", "source.pdf", tmp_path, converter=converter
    )
    cached_text, cached_metrics = extract_document(
        "example", "source.pdf", tmp_path, converter=converter
    )

    assert text == "# Extracted document"
    assert cached_text == text
    assert converter.sources == [Path("source.pdf")]
    assert first_metrics["extract_cached"] is False
    assert cached_metrics["extract_cached"] is True
    assert cached_metrics["extract_seconds"] == 0.0


def test_rejects_empty_extracted_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No text was extracted"):
        extract_document(
            "empty",
            "source.pdf",
            tmp_path,
            converter=_FakeConverter("  \n"),
        )
