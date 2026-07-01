from __future__ import annotations

from pathlib import Path
import time

from .cache import AnswerCache
from .config import AppConfig
from .evaluation import ragas_style_scores
from .llm import Answerer, build_answerer
from .models import Chunk
from .parsers import DocumentParser
from .store import HybridIndex, ManifestStore, file_hash
from .text_utils import chunk_text, stable_hash


class RAGPipeline:
    def __init__(self, config: AppConfig, answerer: Answerer | None = None):
        self.config = config
        self.parser = DocumentParser()
        self.manifest = ManifestStore(config.manifest_path)
        self.index = HybridIndex(config, self.manifest)
        self.cache = AnswerCache(config.cache_dir)
        self.answerer = answerer if answerer is not None else build_answerer(config)

    def ingest_path(self, path: Path | str) -> dict:
        target = Path(path)
        docs = self.parser.parse_path(target)
        indexed_files = 0
        skipped_files = 0
        indexed_chunks = 0

        for doc in docs:
            source = doc.source_path
            digest = file_hash(source)
            if self.manifest.file_seen(source, digest):
                skipped_files += 1
                continue
            self.index.remove_file(source)
            chunks = [
                Chunk(
                    id=stable_hash(f"{source.resolve()}:{idx}:{text}"),
                    text=text,
                    source=str(source),
                    modality=doc.modality,
                    metadata={**doc.metadata, "filename": source.name, "hash": digest},
                )
                for idx, text in enumerate(chunk_text(doc.content, self.config.chunk_size, self.config.chunk_overlap))
            ]
            self.index.add_chunks(chunks)
            self.manifest.mark_file(source, digest)
            self.manifest.save()
            indexed_files += 1
            indexed_chunks += len(chunks)

        if indexed_chunks:
            self.cache.clear()

        return {
            "indexed_files": indexed_files,
            "skipped_files": skipped_files,
            "indexed_chunks": indexed_chunks,
            "total_chunks": len(self.manifest.chunks()),
        }

    def ingest_upload(self, filename: str, content: bytes) -> dict:
        safe_name = Path(filename).name
        destination = self.config.upload_dir / safe_name
        destination.write_bytes(content)
        return self.ingest_path(destination)

    def list_files(self) -> list[dict]:
        chunks = self.manifest.chunks()
        items = []
        for source, digest in sorted(self.manifest.data.get("files", {}).items()):
            source_path = Path(source)
            source_chunks = [chunk for chunk in chunks if Path(chunk.source).resolve() == source_path.resolve()]
            modalities = sorted({chunk.modality for chunk in source_chunks})
            items.append(
                {
                    "id": stable_hash(source),
                    "source": source,
                    "filename": source_path.name,
                    "hash": digest,
                    "exists": source_path.exists(),
                    "chunks_count": len(source_chunks),
                    "modalities": modalities,
                }
            )
        return items

    def _source_for_file_id(self, file_id: str) -> Path:
        for source in self.manifest.data.get("files", {}):
            if stable_hash(source) == file_id:
                return Path(source)
        raise FileNotFoundError(f"file id not found: {file_id}")

    def file_detail(self, file_id: str) -> dict:
        source_path = self._source_for_file_id(file_id)
        chunks = [
            chunk
            for chunk in self.manifest.chunks()
            if Path(chunk.source).resolve() == source_path.resolve()
        ]
        digest = self.manifest.data.get("files", {}).get(str(source_path.resolve()), "")
        return {
            "id": file_id,
            "source": str(source_path),
            "filename": source_path.name,
            "hash": digest,
            "exists": source_path.exists(),
            "chunks_count": len(chunks),
            "modalities": sorted({chunk.modality for chunk in chunks}),
            "chunks": [
                {
                    "id": chunk.id,
                    "modality": chunk.modality,
                    "preview": chunk.text[:360],
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }

    def delete_file(self, file_id: str) -> dict:
        source_path = self._source_for_file_id(file_id)
        filename = source_path.name
        self.index.remove_file(source_path)
        if source_path.exists() and self.config.upload_dir in source_path.resolve().parents:
            source_path.unlink()
        self.cache.clear()
        return {"status": "deleted", "deleted_file": filename, "id": file_id}

    def answer(self, question: str, limit: int = 5) -> dict:
        cached = self.cache.get(question)
        if cached:
            return cached
        started = time.perf_counter()
        results = self.index.search(question, limit=limit)
        contexts = [item.chunk.text for item in results]
        answer, provider, generation_error = self._generate_answer(question, contexts)
        metrics = ragas_style_scores(question, answer, contexts)
        payload = {
            "question": question,
            "answer": answer,
            "answer_provider": provider,
            "generation_error": generation_error,
            "sources": [
                {
                    "source": item.chunk.source,
                    "filename": item.chunk.metadata.get("filename", Path(item.chunk.source).name),
                    "modality": item.chunk.modality,
                    "score": round(item.score, 4),
                    "bm25_score": round(item.bm25_score, 4),
                    "vector_score": round(item.vector_score, 4),
                    "preview": item.chunk.text[:220],
                }
                for item in results
            ],
            "metrics": metrics,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "cached": False,
        }
        self.cache.set(question, payload)
        return payload

    def evaluate(self, question: str, ground_truth: str) -> dict[str, float]:
        response = self.answer(question)
        return ragas_style_scores(
            question=question,
            answer=response["answer"],
            contexts=[source["preview"] for source in response["sources"]],
            ground_truth=ground_truth,
        )

    def stats(self) -> dict:
        chunks = self.manifest.chunks()
        files = self.manifest.data.get("files", {})
        return {
            "files": len(files),
            "chunks": len(chunks),
            "modalities": sorted({chunk.modality for chunk in chunks}),
            "cache_files": len(list(self.config.cache_dir.glob("*.json"))),
        }

    def reset(self) -> None:
        self.index.reset()
        self.cache.clear()

    def _generate_answer(self, question: str, contexts: list[str]) -> tuple[str, str, str | None]:
        if not contexts:
            return self._compose_answer(question, contexts), "local-template", None
        if self.answerer:
            try:
                answer = self.answerer.generate(question, contexts)
                if answer:
                    return answer, self.answerer.provider_name, None
            except Exception as exc:
                return self._compose_answer(question, contexts), "local-fallback", str(exc)
        return self._compose_answer(question, contexts), "local-template", None

    def _compose_answer(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "知识库中暂未检索到足够相关内容，请先上传 PDF、Markdown 或图片资料后再提问。"
        bullets = []
        for idx, context in enumerate(contexts[:3], start=1):
            compact = context.replace("\n", " ")
            bullets.append(f"{idx}. {compact[:180]}")
        return f"根据知识库检索结果，问题「{question}」可以这样回答：\n" + "\n".join(bullets)
