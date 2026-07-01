from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import AppConfig
from .pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
pipeline = RAGPipeline(AppConfig.from_root(PROJECT_ROOT))

app = FastAPI(title="Personal Multimodal RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class EvaluateRequest(BaseModel):
    question: str
    ground_truth: str


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> dict:
    return pipeline.stats()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    return pipeline.ingest_upload(file.filename or "upload.bin", content)


@app.get("/api/files")
def files() -> dict:
    return {"files": pipeline.list_files()}


@app.get("/api/files/{file_id}")
def file_detail(file_id: str) -> dict:
    try:
        return pipeline.file_detail(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str) -> dict:
    try:
        return pipeline.delete_file(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/ingest-path")
def ingest_path(path: str = Form(...)) -> dict:
    return pipeline.ingest_path(Path(path))


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    return pipeline.answer(request.question)


@app.post("/api/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    return pipeline.evaluate(request.question, request.ground_truth)


@app.post("/api/reset")
def reset() -> dict:
    pipeline.reset()
    return {"status": "reset"}
