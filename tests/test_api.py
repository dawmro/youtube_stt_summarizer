"""
Pytest suite for FastAPI Headless Mode (api.py)
Run with:
    pytest tests/test_api.py -v
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import FastAPI app and pipeline types
from api import app
from app_c1 import (
    SttUpdate, SummaryUpdate, TranscriptSegment,
    HybridDoc, SourceRef
)

# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_lifecycle():
    """Prevent actual Whisper/Ollama/DB initialization during tests.
    
    Design Note:
    PATHS is a @dataclass(frozen=True). Patching individual attributes like 
    PATHS.ensure triggers FrozenInstanceError during mock teardown. 
    We patch the entire api.PATHS reference with a MagicMock instead.
    """
    mock_runtime = MagicMock()
    mock_runtime.llm = MagicMock()
    mock_runtime.embeddings = MagicMock()
    mock_runtime.qa_chain = MagicMock()
    mock_runtime.qa_chain.prompt.template = "Test {context} {question} {chat_history}"

    mock_paths = MagicMock()

    with patch("api.PATHS", mock_paths), \
         patch("api.init_db"), \
         patch("api.build_runtime", return_value=mock_runtime), \
         patch("api.runtime", mock_runtime):
        yield mock_runtime


@pytest.fixture
def client():
    """Yields a synchronous TestClient with lifespan triggered."""
    with TestClient(app) as c:
        yield c


# ── Health Endpoint ──────────────────────────────────────────────────────────
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert "vector_db" in data
    assert "whisper_model" in data
    assert "llm_model" in data


# ── Streaming: /transcribe ───────────────────────────────────────────────────
def test_transcribe_streaming(client):
    mock_updates = [
        SttUpdate(message="⬇️ Downloading audio...", transcript=None, segments=None, progress=10),
        SttUpdate(message="🎙️ Transcribing...", transcript=None, segments=None, progress=55),
        SttUpdate(
            message="✅ Transcript ready",
            transcript="Hello world. This is a test transcript.",
            segments=[TranscriptSegment(segment_id=0, start=0.0, end=2.5, text="Hello world. This is a test transcript.")],
            progress=100
        ),
    ]

    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(mock_updates)):
        response = client.post("/transcribe", json={"video_url": "https://youtube.com/watch?v=test12345AB"})
        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers["content-type"]

        lines = [line for line in response.content.decode().splitlines() if line.strip()]
        assert len(lines) == 3

        for i, line in enumerate(lines):
            payload = json.loads(line)
            assert payload["stage"] == "stt"
            assert payload["progress"] == mock_updates[i].progress
            assert payload["message"] == mock_updates[i].message

        final = json.loads(lines[-1])
        assert "Hello world" in final["transcript"]
        assert len(final["segments"]) == 1
        assert final["segments"][0]["segment_id"] == 0


# ── Streaming: /summarize ────────────────────────────────────────────────────
def test_summarize_streaming(client):
    mock_stt = [
        SttUpdate(
            message="✅ Transcript ready",
            transcript="The speaker explains quantum computing basics.",
            segments=[TranscriptSegment(segment_id=0, start=0.0, end=3.0, text="The speaker explains quantum computing basics.")],
            progress=100
        )
    ]
    mock_summary = [
        SummaryUpdate(message="📝 Generating direct summary...", summary=None, progress=30),
        SummaryUpdate(message="✅ Final summary ready.", summary="Quantum computing uses qubits for parallel processing.", progress=100)
    ]

    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(mock_stt)):
        with patch("api.summarize_transcript_stream", return_value=iter(mock_summary)):
            # ✅ Fixed: 11-char video ID passes require_video_id() validation
            response = client.post("/summarize", json={"video_url": "https://youtube.com/watch?v=test12345AB"})
            assert response.status_code == 200

            lines = [line for line in response.content.decode().splitlines() if line.strip()]
            stages = [json.loads(line)["stage"] for line in lines]
            
            assert "stt" in stages
            assert "summary" in stages

            final = json.loads(lines[-1])
            assert final["stage"] == "summary"
            assert final["progress"] == 100
            assert "qubits" in final["summary"]


# ── Synchronous: /qa (Success Path) ──────────────────────────────────────────
def test_qa_endpoint_success(client):
    mock_stt = [
        SttUpdate(
            message="✅ Transcript ready",
            transcript="The video covers AI safety and alignment protocols.",
            segments=[TranscriptSegment(segment_id=0, start=0.0, end=4.0, text="The video covers AI safety and alignment protocols.")],
            progress=100
        )
    ]
    mock_docs = [
        HybridDoc(
            page_content="The video covers AI safety and alignment protocols.",
            metadata={"start": 0.0, "end": 4.0, "chunk_id": 0, "hybrid_score": 0.92}
        )
    ]
    mock_context = "[S1] The video covers AI safety and alignment protocols."
    mock_sources = {
        "S1": SourceRef(
            start=0.0, end=4.0, label="00:00:00 - 00:00:04",
            url="https://youtube.com/watch?v=test12345AB&t=0s", chunk_id=0, text="..."
        )
    }

    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(mock_stt)):
        with patch("api.get_or_create_chunks"), patch("api.get_or_create_vector_store"):
            with patch("api.hybrid_search", return_value=mock_docs):
                with patch("api.build_context_with_sources", return_value=(mock_context, mock_sources)):
                    with patch("api.run_llm_dynamic", return_value="The speaker discusses AI safety [S1]."):
                        with patch("api.render_clickable_answer", return_value="The speaker discusses AI safety [00:00:00 - 00:00:04](url)."):
                            with patch("api.estimate_tokens", return_value=42):
                                # ✅ Fixed: 11-char video ID
                                response = client.post("/qa", json={
                                    "video_url": "https://youtube.com/watch?v=test12345AB",
                                    "question": "What is the main topic?",
                                    "top_k": 4
                                })
                                assert response.status_code == 200
                                data = response.json()
                                assert "AI safety" in data["answer"]
                                assert len(data["sources"]) == 1
                                assert data["sources"][0]["label"] == "00:00:00 - 00:00:04"
                                assert data["context_tokens"] == 42
                                assert data["hybrid_scores"] == [0.92]


# ── Synchronous: /qa (Empty Context Fallback) ────────────────────────────────
def test_qa_endpoint_no_context(client):
    mock_stt = [
        SttUpdate(message="✅ Ready", transcript="Short clip.", segments=[
            TranscriptSegment(segment_id=0, start=0.0, end=1.0, text="Short clip.")
        ], progress=100)
    ]

    with patch("api.fetch_transcript_from_stt_stream", return_value=iter(mock_stt)):
        with patch("api.get_or_create_chunks"), patch("api.get_or_create_vector_store"):
            with patch("api.hybrid_search", return_value=[]):
                # ✅ Fixed: 11-char video ID
                response = client.post("/qa", json={
                    "video_url": "https://youtube.com/watch?v=test12345AB",
                    "question": "Unknown topic?"
                })
                assert response.status_code == 200
                data = response.json()
                assert "couldn't find relevant transcript evidence" in data["answer"]
                assert data["sources"] == []
                assert data["context_tokens"] == 0
                assert data["hybrid_scores"] == []


# ── Validation & Error Handling ──────────────────────────────────────────────
def test_qa_missing_question_validation(client):
    """Pydantic should reject missing required fields with 422."""
    response = client.post("/qa", json={"video_url": "https://youtube.com/watch?v=test12345AB"})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "question"]
    assert error["type"] == "missing"


def test_transcribe_invalid_url_format(client):
    """API should accept the payload but pipeline mocking handles validation."""
    with patch("api.fetch_transcript_from_stt_stream", return_value=iter([])):
        response = client.post("/transcribe", json={"video_url": "not_a_url"})
        assert response.status_code == 200
        lines = [l for l in response.content.decode().splitlines() if l.strip()]
        assert len(lines) == 0


# ── Optional: Run with real models (Integration Test) ────────────────────────
@pytest.mark.integration
def test_health_with_real_runtime():
    """Skipped by default. Run with: pytest -m integration tests/test_api.py"""
    with TestClient(app) as real_client:
        resp = real_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["models_loaded"] is True