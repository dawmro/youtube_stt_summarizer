"""
test_api.py — FastAPI Headless Mode tests aligned with LangChain 0.3.x.
Mocks ChatOllama & RunnableSequence. Validates NDJSON streaming, Pydantic
contracts, and error routing. Zero heavy dependencies.

Run with: pytest tests/test_api.py -v
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api import app
from app import SttUpdate, SummaryUpdate, TranscriptSegment, HybridDoc, SourceRef

@pytest.fixture(autouse=True)
def mock_lifecycle():
    """Stub runtime, CFG, and DB for fast, isolated API tests."""
    mock_runtime = MagicMock()
    
    # LangChain 0.3.x: ChatOllama & RunnableSequence patterns
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Mocked LLM response")
    mock_runtime.llm = mock_llm

    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.1] * 768
    mock_emb.embed_documents.return_value = [[0.1] * 768]
    mock_runtime.embeddings = mock_emb

    mock_runtime.whisper = MagicMock()

    mock_qa_chain = MagicMock()
    mock_qa_chain.prompt.template = "Test {context} {question} {chat_history}"
    mock_qa_chain.invoke.return_value = MagicMock(content="Mocked QA answer")
    mock_runtime.qa_chain = mock_qa_chain

    mock_cfg = MagicMock()
    mock_cfg.vector_db_type = "qdrant"
    mock_cfg.retrieval_top_k = 4
    mock_cfg.hybrid_top_k_candidates = 8
    mock_cfg.hybrid_dense_weight = 0.7
    mock_cfg.whisper_model_size = "small"
    mock_cfg.llm_model = "llama3.1:8b-instruct-q8_0"

    with patch("api.PATHS", MagicMock()), \
         patch("api.CFG", mock_cfg), \
         patch("api.init_db"), \
         patch("api.build_runtime", return_value=mock_runtime), \
         patch("api.runtime", mock_runtime):
        yield mock_runtime

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# ── Health ───────────────────────────────────────────────────────────────────
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["models_loaded"] is True

# ── Streaming: /transcribe ───────────────────────────────────────────────────
def test_transcribe_streaming(client):
    updates = [
        SttUpdate("⬇️ Downloading...", None, None, 10),
        SttUpdate("✅ Ready", "Hello world.", [TranscriptSegment(0, 0.0, 2.5, "Hello world.")], 100)
    ]
    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(updates)):
        r = client.post("/transcribe", json={"video_url": "https://youtube.com/watch?v=test12345AB"})
        assert r.status_code == 200 and "application/x-ndjson" in r.headers["content-type"]
        lines = [l for l in r.content.decode().splitlines() if l.strip()]
        assert len(lines) == 2
        final = json.loads(lines[-1])
        assert "Hello world" in final["transcript"]

# ── Streaming: /summarize ────────────────────────────────────────────────────
def test_summarize_streaming(client):
    stt = [SttUpdate("✅ Ready", "Quantum basics.", [TranscriptSegment(0, 0.0, 3.0, "Quantum basics.")], 100)]
    summ = [SummaryUpdate("📝 Generating...", None, 30), SummaryUpdate("✅ Done.", "Uses qubits.", 100)]
    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(stt)):
        with patch("api.summarize_transcript_stream", return_value=iter(summ)):
            r = client.post("/summarize", json={"video_url": "https://youtube.com/watch?v=test12345AB"})
            assert r.status_code == 200
            stages = [json.loads(l)["stage"] for l in r.content.decode().splitlines() if l.strip()]
            assert "stt" in stages and "summary" in stages

# ── Synchronous: /qa (Success) ───────────────────────────────────────────────
def test_qa_endpoint_success(client):
    stt = [SttUpdate("✅ Ready", "AI safety.", [TranscriptSegment(0, 0.0, 4.0, "AI safety.")], 100)]
    docs = [HybridDoc("AI safety.", {"start": 0.0, "end": 4.0, "chunk_id": 0, "video_id": "test12345AB", "segment_ids": [0], "hybrid_score": 0.92})]
    ctx = "[S1] (Video: test12345AB) AI safety."
    src = {"S1": SourceRef(0.0, 4.0, "00:00:00 - 00:00:04", "https://youtube.com/watch?v=test12345AB&t=0s", 0, "...")}

    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(stt)):
        with patch("api.get_or_create_chunks"), patch("api.get_or_create_vector_store"):
            with patch("api.hybrid_search", return_value=docs):
                with patch("api.build_context_with_sources", return_value=(ctx, src)):
                    with patch("api.run_llm_dynamic", return_value="Discusses AI safety [S1]."):
                        with patch("api.render_clickable_answer", return_value="Discusses AI safety [00:00:00 - 00:00:04](url)."):
                            with patch("api.estimate_tokens", return_value=42):
                                r = client.post("/qa", json={"video_url": "https://youtube.com/watch?v=test12345AB", "question": "Topic?", "top_k": 4})
                                assert r.status_code == 200
                                d = r.json()
                                assert "AI safety" in d["answer"] and d["context_tokens"] == 42

# ── Synchronous: /qa (Empty Context) ─────────────────────────────────────────
def test_qa_endpoint_no_context(client):
    stt = [SttUpdate("✅ Ready", "Short.", [TranscriptSegment(0, 0.0, 1.0, "Short.")], 100)]
    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(stt)):
        with patch("api.get_or_create_chunks"), patch("api.get_or_create_vector_store"):
            with patch("api.hybrid_search", return_value=[]):
                r = client.post("/qa", json={"video_url": "https://youtube.com/watch?v=test12345AB", "question": "Unknown?"})
                assert r.status_code == 200
                assert "couldn't find relevant transcript evidence" in r.json()["answer"]

# ── Validation ───────────────────────────────────────────────────────────────
def test_qa_missing_question(client):
    r = client.post("/qa", json={"video_url": "https://youtube.com/watch?v=test12345AB"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "question"]

# ── Integration (Skipped by default) ─────────────────────────────────────────
@pytest.mark.integration
def test_health_with_real_runtime():
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200