"""
FastAPI Headless Mode for YouTube STT Summarizer & Q&A

Design Rationale:
- Stateless per-request architecture: no shared Gradio state, safe for concurrent clients.
- Reuses 100% of existing pipeline functions from app.py (STT, summarization, hybrid search, citation rendering).
- NDJSON streaming for long-running tasks (transcription, summarization) to avoid HTTP timeouts.
- Modern FastAPI lifespan context manager for safe startup/shutdown.
- Pydantic models for strict request/response validation and auto-generated OpenAPI docs.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import asyncio
import logging

# Import core pipeline components from the main application
from app import (
    build_runtime, PATHS, init_db, CFG,
    fetch_transcript_from_stt_stream, summarize_transcript_stream,
    SessionState, get_or_create_chunks, get_or_create_vector_store,
    hybrid_search, build_context_with_sources, render_clickable_answer,
    run_llm_dynamic, estimate_tokens
)

logger = logging.getLogger("api")
runtime = None


# ── Lifecycle Management ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and cache on startup, clean up on shutdown."""
    global runtime
    logger.info("Initializing API runtime (Whisper + Ollama + Vector DB)...")
    PATHS.ensure()
    init_db()
    runtime = build_runtime()
    logger.info("API runtime ready. Endpoints available.")
    yield
    logger.info("Shutting down API runtime.")


app = FastAPI(
    title="YouTube STT Summarizer API",
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow external UIs (React, Vue, mobile apps) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    vector_db: str
    whisper_model: str
    llm_model: str


class TranscribeRequest(BaseModel):
    video_url: str


class SummarizeRequest(BaseModel):
    video_url: str
    prompt_override: Optional[str] = Field(None, description="Custom prompt template. Use {transcript}, {chunk}, or {summaries}.")


class QARequest(BaseModel):
    video_url: str
    question: str
    top_k: int = Field(default=CFG.retrieval_top_k, ge=1, le=10)
    prompt_override: Optional[str] = Field(None, description="Custom QA prompt. Use {context}, {question}, {chat_history}.")


class SourceRefResponse(BaseModel):
    start: float
    end: float
    label: str
    url: str
    chunk_id: Optional[int]
    text: str


class QAResponse(BaseModel):
    answer: str
    sources: List[SourceRefResponse]
    context_tokens: int
    hybrid_scores: List[float]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """Check if models are loaded and the API is ready."""
    if not runtime:
        raise HTTPException(503, "Runtime not initialized yet")
    return HealthResponse(
        status="ok",
        models_loaded=True,
        vector_db=CFG.vector_db_type,
        whisper_model=CFG.whisper_model_size,
        llm_model=CFG.llm_model,
    )


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    """Stream YouTube transcription progress and final result as NDJSON."""
    if not runtime:
        raise HTTPException(503, "Runtime not initialized")

    async def stream():
        try:
            for update in fetch_transcript_from_stt_stream(req.video_url, runtime):
                payload = {
                    "stage": "stt",
                    "progress": update.progress,
                    "message": update.message,
                }
                if update.transcript:
                    payload["transcript"] = update.transcript
                if update.segments:
                    payload["segments"] = [s.to_dict() for s in update.segments]
                yield json.dumps(payload) + "\n"
                await asyncio.sleep(0)  # Yield to event loop to prevent blocking
        except Exception as exc:
            yield json.dumps({"stage": "stt", "error": str(exc), "progress": 0}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Stream transcript fetch + hierarchical summarization as NDJSON."""
    if not runtime:
        raise HTTPException(503, "Runtime not initialized")

    async def stream():
        try:
            state = SessionState()
            transcript = None
            segments = None

            # Phase 1: Transcription
            for update in fetch_transcript_from_stt_stream(req.video_url, runtime):
                if update.transcript and update.segments:
                    transcript = update.transcript
                    segments = update.segments
                yield json.dumps({
                    "stage": "stt",
                    "progress": update.progress,
                    "message": update.message
                }) + "\n"
                await asyncio.sleep(0)

            if not transcript or not segments:
                yield json.dumps({"stage": "stt", "error": "Failed to fetch transcript"}) + "\n"
                return

            state.set_transcript(req.video_url, transcript, segments)

            # Phase 2: Summarization
            for update in summarize_transcript_stream(
                state.processed_transcript, runtime, req.prompt_override or ""
            ):
                payload = {
                    "stage": "summary",
                    "progress": update.progress,
                    "message": update.message,
                }
                if update.summary:
                    payload["summary"] = update.summary
                yield json.dumps(payload) + "\n"
                await asyncio.sleep(0)
        except Exception as exc:
            yield json.dumps({"stage": "pipeline", "error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    """Stateless timestamp-aware Q&A with hybrid retrieval and citation rendering."""
    if not runtime:
        raise HTTPException(503, "Runtime not initialized")

    try:
        state = SessionState()

        # Fetch or load cached transcript
        transcript = None
        segments = None
        for update in fetch_transcript_from_stt_stream(req.video_url, runtime):
            if update.transcript and update.segments:
                transcript = update.transcript
                segments = update.segments
        if not transcript or not segments:
            raise HTTPException(400, "Failed to fetch or cache transcript")

        state.set_transcript(req.video_url, transcript, segments)
        get_or_create_chunks(state, runtime)
        vector_store = get_or_create_vector_store(state, runtime)

        # Hybrid dense + BM25 retrieval
        docs = hybrid_search(
            question=req.question,
            vector_store=vector_store,
            chunks=state.chunks,
            embeddings=runtime.embeddings,
            top_k=req.top_k,
        )

        if not docs:
            return QAResponse(
                answer="I couldn't find relevant transcript evidence for that question.",
                sources=[],
                context_tokens=0,
                hybrid_scores=[],
            )

        context, source_lookup = build_context_with_sources(docs, state.video_id)
        context_tokens = estimate_tokens(context)

        # Generate answer (stateless → no chat history)
        tpl = req.prompt_override or runtime.qa_chain.prompt.template
        raw_answer = run_llm_dynamic(runtime.llm, tpl, {
            "context": context,
            "question": req.question,
            "chat_history": "None",
        })

        rendered_answer = render_clickable_answer(raw_answer, source_lookup)
        sources = [SourceRefResponse(**ref.__dict__) for ref in source_lookup.values()]
        scores = [doc.metadata.get("hybrid_score", 0.0) for doc in docs]

        return QAResponse(
            answer=rendered_answer,
            sources=sources,
            context_tokens=context_tokens,
            hybrid_scores=scores,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("QA pipeline error")
        raise HTTPException(500, str(exc))