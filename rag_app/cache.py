from __future__ import annotations

from pathlib import Path
import json

from .text_utils import stable_hash


class AnswerCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, question: str) -> Path:
        return self.cache_dir / f"{stable_hash(question)[:24]}.json"

    def get(self, question: str) -> dict | None:
        path = self._path(question)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cached"] = True
        return payload

    def set(self, question: str, payload: dict) -> None:
        path = self._path(question)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
