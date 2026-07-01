from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

from .config import AppConfig
from .pipeline import RAGPipeline


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Personal multimodal RAG knowledge base")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("path")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    sub.add_parser("stats")
    args = parser.parse_args()

    pipeline = RAGPipeline(AppConfig.from_root(Path.cwd()))
    if args.command == "ingest":
        result = pipeline.ingest_path(Path(args.path))
    elif args.command == "ask":
        result = pipeline.answer(args.question)
    else:
        result = pipeline.stats()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
