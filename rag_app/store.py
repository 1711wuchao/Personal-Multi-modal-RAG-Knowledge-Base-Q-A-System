from __future__ import annotations

from pathlib import Path
import json
import os

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

from .config import AppConfig
from .models import Chunk, SearchResult
from .text_utils import stable_hash, tokenize


class ManifestStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"files": {}, "chunks": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.data = {"files": {}, "chunks": []}
        self.save()

    def file_seen(self, path: Path, file_hash: str) -> bool:
        return self.data["files"].get(str(path.resolve())) == file_hash

    def mark_file(self, path: Path, file_hash: str) -> None:
        self.data["files"][str(path.resolve())] = file_hash

    def remove_file(self, path: Path) -> None:
        source = str(path)
        resolved = str(path.resolve())
        for key in [source, resolved]:
            self.data["files"].pop(key, None)

    def remove_file_chunks(self, path: Path) -> list[str]:
        source = str(path)
        resolved = str(path.resolve())
        removed: list[str] = []
        kept = []
        for item in self.data["chunks"]:
            if item["source"] in {source, resolved}:
                removed.append(item["id"])
            else:
                kept.append(item)
        self.data["chunks"] = kept
        return removed

    def add_chunks(self, chunks: list[Chunk]) -> None:
        existing = {item["id"] for item in self.data["chunks"]}
        for chunk in chunks:
            if chunk.id in existing:
                continue
            self.data["chunks"].append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "modality": chunk.modality,
                    "metadata": chunk.metadata,
                }
            )

    def chunks(self) -> list[Chunk]:
        return [
            Chunk(
                id=item["id"],
                text=item["text"],
                source=item["source"],
                modality=item.get("modality", "text"),
                metadata=item.get("metadata", {}),
            )
            for item in self.data["chunks"]
        ]


class HybridIndex:
    collection_name = "personal_knowledge"

    def __init__(self, config: AppConfig, manifest: ManifestStore):
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        self.config = config
        self.manifest = manifest
        self.vectorizer = HashingVectorizer(tokenizer=tokenize, alternate_sign=False, n_features=2048, norm="l2")
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.manifest.add_chunks(chunks)
        vectors = self.vectorizer.transform([chunk.text for chunk in chunks]).toarray().tolist()
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=vectors,
            metadatas=[
                {
                    "source": chunk.source,
                    "modality": chunk.modality,
                    **{f"meta_{key}": value for key, value in chunk.metadata.items()},
                }
                for chunk in chunks
            ],
        )
        self.manifest.save()

    def remove_file(self, path: Path) -> None:
        removed_ids = self.manifest.remove_file_chunks(path)
        if removed_ids:
            self.collection.delete(ids=removed_ids)
        self.manifest.remove_file(path)
        self.manifest.save()

    def reset(self) -> None:
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.manifest.clear()
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        chunks = self.manifest.chunks()
        if not chunks:
            return []

        corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(tokenize(query))

        matrix = self.vectorizer.transform([chunk.text for chunk in chunks])
        query_vector = self.vectorizer.transform([query])
        vector_scores = cosine_similarity(query_vector, matrix).flatten()

        results: list[SearchResult] = []
        max_bm25 = max(float(score) for score in bm25_scores) or 1.0
        for idx, chunk in enumerate(chunks):
            bm25_score = float(bm25_scores[idx]) / max_bm25
            vector_score = float(vector_scores[idx])
            score = 0.58 * bm25_score + 0.42 * vector_score
            if score <= 0:
                continue
            results.append(SearchResult(chunk=chunk, bm25_score=bm25_score, vector_score=vector_score, score=score))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]


def file_hash(path: Path) -> str:
    return stable_hash(path.read_bytes())
