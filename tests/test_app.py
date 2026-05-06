"""
test_app.py — Isolated unit tests for pure utilities, data models, chunking,
regex rendering, and session-state transitions.

LangChain 0.3.x Alignment:
- No deprecated imports. Pure functions only.
- build_retrieval_chunks requires an embeddings argument (mocked).
- Context builder prefixes lines with (Video: <id>).
- SessionState uses chat_history instead of last_question/last_answer.

Run with: pytest tests/test_app.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app  # noqa: E402

@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    emb.embed_query.return_value = [0.1] * 768
    return emb

# ===========================================================================
# 1. URL / video-id utilities
# ===========================================================================
class TestGetVideoId:
    def test_standard_watch_url(self):
        assert app.get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    def test_short_youtu_be(self):
        assert app.get_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    def test_shorts_url(self):
        assert app.get_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    def test_url_with_extra_params(self):
        assert app.get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123") == "dQw4w9WgXcQ"
    def test_invalid_url_returns_none(self):
        assert app.get_video_id("https://example.com/not-a-video") is None
    def test_empty_string_returns_none(self):
        assert app.get_video_id("") is None

class TestRequireVideoId:
    def test_raises_on_invalid_url(self):
        with pytest.raises(ValueError, match="Invalid YouTube URL"):
            app.require_video_id("https://example.com/foo")
    def test_returns_id_on_valid_url(self):
        assert app.require_video_id("https://youtu.be/abcdefghijk") == "abcdefghijk"

# ===========================================================================
# 2. Time / URL formatting
# ===========================================================================
class TestSecondsToHhmmss:
    def test_zero(self): assert app.seconds_to_hhmmss(0) == "00:00:00"
    def test_one_minute(self): assert app.seconds_to_hhmmss(60) == "00:01:00"
    def test_one_hour(self): assert app.seconds_to_hhmmss(3600) == "01:00:00"
    def test_mixed(self): assert app.seconds_to_hhmmss(3725) == "01:02:05"
    def test_negative_clamps_to_zero(self): assert app.seconds_to_hhmmss(-5) == "00:00:00"
    def test_float_truncated(self): assert app.seconds_to_hhmmss(90.9) == "00:01:30"

class TestBuildYoutubeTimeUrl:
    def test_basic(self):
        assert app.build_youtube_time_url("abc123", 125.7) == "https://www.youtube.com/watch?v=abc123&t=125s"
    def test_negative_clamps(self):
        assert app.build_youtube_time_url("abc123", -10) == "https://www.youtube.com/watch?v=abc123&t=0s"

# ===========================================================================
# 3. Hash functions
# ===========================================================================
class TestHashing:
    def test_text_hash_deterministic(self):
        assert app.text_hash("hello") == app.text_hash("hello")
    def test_text_hash_differs(self):
        assert app.text_hash("hello") != app.text_hash("world")
    def test_text_hash_length(self):
        assert len(app.text_hash("x")) == 16
    def test_stable_hash_obj_order_independent(self):
        assert app.stable_hash_obj({"a": 1, "b": 2}) == app.stable_hash_obj({"b": 2, "a": 1})

# ===========================================================================
# 4. Data-model roundtrips
# ===========================================================================
def _make_segment_dict(sid=0, start=1.0, end=5.0, text="Hello world") -> Dict[str, Any]:
    return {"segment_id": sid, "start": start, "end": end, "text": text,
            "words": [{"word": "Hello", "start": 1.0, "end": 1.5, "probability": 0.99}]}

class TestTranscriptSegmentRoundtrip:
    def test_roundtrip(self):
        seg = app.TranscriptSegment.from_dict(_make_segment_dict())
        d = seg.to_dict()
        assert d["segment_id"] == 0 and d["text"] == "Hello world" and len(d["words"]) == 1
    def test_missing_words_defaults(self):
        data = _make_segment_dict(); del data["words"]
        assert app.TranscriptSegment.from_dict(data).words == []

class TestRetrievalChunkRoundtrip:
    def test_roundtrip(self):
        data = {"chunk_id": 0, "text": "txt", "start": 0.0, "end": 10.0, "segment_ids": [0, 1]}
        chunk = app.RetrievalChunk.from_dict(data)
        assert chunk.to_dict()["segment_ids"] == [0, 1]

# ===========================================================================
# 5. Retrieval chunking
# ===========================================================================
def _make_segments(texts: List[str], offset=0.0) -> List[app.TranscriptSegment]:
    segs = []
    t = offset
    for i, txt in enumerate(texts):
        segs.append(app.TranscriptSegment(segment_id=i, start=t, end=t+1.0, text=txt))
        t += 1.0
    return segs

class TestBuildRetrievalChunks:
    def test_empty(self, mock_embeddings):
        assert app.build_retrieval_chunks([], mock_embeddings) == []
    def test_single(self, mock_embeddings):
        chunks = app.build_retrieval_chunks(_make_segments(["Hi"]), mock_embeddings)
        assert len(chunks) == 1 and chunks[0].text == "Hi"
    def test_sequential_ids(self, mock_embeddings):
        chunks = app.build_retrieval_chunks(_make_segments([f"s{i}" for i in range(5)]), mock_embeddings)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

# ===========================================================================
# 6. Q&A rendering
# ===========================================================================
def _make_doc(start, end, text, chunk_id=0, video_id="vid123"):
    doc = MagicMock()
    doc.metadata = {"start": start, "end": end, "chunk_id": chunk_id, "video_id": video_id}
    doc.page_content = text
    return doc

class TestBuildContextWithSources:
    def test_lookup_and_prefix(self):
        docs = [_make_doc(0.0, 10.0, "First chunk")]
        ctx, lookup = app.build_context_with_sources(docs, "vid123")
        assert "S1" in lookup and "(Video: vid123)" in ctx
    def test_empty(self):
        assert app.build_context_with_sources([], "v") == ("", {})

class TestRenderClickableAnswer:
    def _lookup(self):
        return {"S1": app.SourceRef(60.0, 90.0, "00:01:00 - 00:01:30",
                                    "https://youtube.com/watch?v=x&t=60s", 0, "txt")}
    def test_single_replaced(self):
        rendered = app.render_clickable_answer("See [S1].", self._lookup())
        assert "00:01:00 - 00:01:30" in rendered and "[S1]" not in rendered
    def test_unknown_kept(self):
        assert "[S99]" in app.render_clickable_answer("See [S99].", self._lookup())
    def test_references_section(self):
        rendered = app.render_clickable_answer("Answer [S1].", self._lookup())
        assert "**References**" in rendered

# ===========================================================================
# 7. SessionState transitions
# ===========================================================================
class TestSessionState:
    def test_needs_refresh_empty(self):
        assert app.SessionState().needs_transcript_refresh("https://youtu.be/abcdefghijk") is True
    def test_same_video_no_refresh(self):
        st = app.SessionState(video_id="abcdefghijk", processed_transcript="txt", transcript_hash="h")
        assert st.needs_transcript_refresh("https://youtu.be/abcdefghijk") is False
    def test_set_transcript_resets_history(self):
        st = app.SessionState(chat_history=[{"role":"user","content":"q"}], chunks=[MagicMock()])
        st.set_transcript("https://youtu.be/abcdefghijk", "new txt", [])
        assert st.chat_history == [] and st.chunks is None
    def test_gradio_roundtrip(self):
        st = app.SessionState(video_id="v1", processed_transcript="t", chat_history=[])
        restored = app.SessionState.from_gradio(st.to_gradio())
        assert restored.video_id == "v1" and restored.chat_history == []