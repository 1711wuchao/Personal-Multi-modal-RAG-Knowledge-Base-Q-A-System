from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ParsedDocument:
    source_path: Path
    content: str
    modality: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    modality: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    bm25_score: float
    vector_score: float
    score: float

