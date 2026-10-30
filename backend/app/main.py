"""FastAPI application: chat (sync + SSE streaming), ingestion, health and metrics."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.agent.prompts import PROMPT_VERSION
from app.agent.service import run_chat, stream_chat
from app.agent.tools import TOOLS
from app.config import BACKEND_DIR, get_settings
from app.deps import get_store
from app.ingestion.service import ingest_bytes, ingest_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    store = get_store()
    if s.auto_ingest_on_startup and store.count() == 0:
        try:
            ingest_directory(store, s.docs_dir)
        except Exception as exc:  # e.g. Ollama not running yet
            log.warning("Startup ingestion skipped: %s", exc)
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    history: list[Turn] = []


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: str = Field("hybrid", pattern="^(hybrid|dense|sparse)$")


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_provider": s.llm_provider,
        "chat_model": "offline" if s.llm_provider == "offline" else s.chat_model,
        "embedding_model": "hashing" if s.llm_provider == "offline" else s.embedding_model,
        "vector_store": "qdrant-server" if s.qdrant_url else "qdrant-embedded",
        "chunks_indexed": get_store().count(),
        "prompt_version": PROMPT_VERSION,
        "tools": [t.name for t in TOOLS],
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    return run_chat(req.question, [t.model_dump() for t in req.history])


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_chat(req.question, [t.model_dump() for t in req.history]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sk/chat")
async def sk_chat_endpoint(req: ChatRequest) -> dict:
    """Same assistant orchestrated by Semantic Kernel auto function calling."""
    from app.sk.orchestrator import sk_chat

    try:
        return await sk_chat(req.question, [t.model_dump() for t in req.history])
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    """Debug endpoint: inspect raw retrieval for a query in any mode."""
    hits = get_store().search(req.query, k=req.k, mode=req.mode)  # type: ignore[arg-type]
    return {"query": req.query, "mode": req.mode, "hits": [h.to_dict() for h in hits]}


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": [{"source": k, "chunks": v} for k, v in get_store().list_sources().items()]}


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    results = {}
    for f in files:
        try:
            results[f.filename] = ingest_bytes(get_store(), await f.read(), f.filename or "upload.txt")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ingested": results}


@app.post("/api/documents/reindex")
def reindex() -> dict:
    store = get_store()
    store.reset()
    return {"ingested": ingest_directory(store, get_settings().docs_dir)}


@app.delete("/api/documents/{source}")
def delete_document(source: str) -> dict:
    get_store().delete_source(source)
    return {"deleted": source}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Serve the built UI when it exists, so the whole app can run on one port without Node.
UI_DIST = BACKEND_DIR.parent / "frontend" / "dist"
if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
