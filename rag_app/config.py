from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    root: Path
    data_dir: Path
    upload_dir: Path
    chroma_dir: Path
    cache_dir: Path
    manifest_path: Path
    chunk_size: int = 420
    chunk_overlap: int = 80
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    @classmethod
    def from_root(cls, root: Path | str) -> "AppConfig":
        base = Path(root).resolve()
        load_dotenv(base / ".env")
        data_dir = base / ".rag_data"
        config = cls(
            root=base,
            data_dir=data_dir,
            upload_dir=data_dir / "uploads",
            chroma_dir=data_dir / "chroma",
            cache_dir=data_dir / "cache",
            manifest_path=data_dir / "manifest.json",
            llm_api_key=os.getenv("OPENAI_API_KEY", ""),
            llm_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
        for path in [config.data_dir, config.upload_dir, config.chroma_dir, config.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)
        return config
