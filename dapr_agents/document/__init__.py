from typing import TYPE_CHECKING

from .embedder import NVIDIAEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
from .fetcher import ArxivFetcher
from .reader import PyMuPDFReader, PyPDFReader

if TYPE_CHECKING:
    from .splitter import TextSplitter

__all__ = [
    "ArxivFetcher",
    "PyMuPDFReader",
    "PyPDFReader",
    "TextSplitter",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "NVIDIAEmbedder",
]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "TextSplitter":
        from .splitter import TextSplitter

        return TextSplitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
