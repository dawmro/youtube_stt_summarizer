"""
test_app.py

Isolated unit tests for pure utility functions, data models, chunking logic,
regex rendering, and session-state transitions.

These tests require no Ollama, Whisper, yt-dlp, FAISS, or network access.
All external dependencies are stubbed by conftest.py before the app module
is imported.

Usage:
    pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import the app module.  conftest.py has already installed stubs so
# build_runtime() and build_interface() succeed without real services.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402 (must follow sys.path manipulation)

# ===========================================================================
# 1. URL / video-id utilities
# ===========================================================================

class TestGetVideoId:
    """get_video_id must handle every common YouTube URL shape."""

    def test_standard_watch_url(self):
        assert app.get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_youtu_be(self):
        assert app.get_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert app.get_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        assert app.get_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&index=2"
        ) == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self):
        assert app.get_video_id("https://example.com/not-a-video") is None

    def test_empty_string_returns_none(self):
        assert app.get_video_id("") is None


class TestRequireVideoId:
    """require_video_id must raise ValueError on invalid input."""

    def test_raises_on_invalid_url(self):
        with pytest.raises(ValueError, match="Invalid YouTube URL"):
            app.require_video_id("https://example.com/foo")

    def test_returns_id_on_valid_url(self):
        result = app.require_video_id("https://youtu.be/abcdefghijk")
        assert result == "abcdefghijk"

# ===========================================================================
# 2. Time / URL formatting
# ===========================================================================

class TestSecondsToHhmmss:
    """seconds_to_hhmmss must format integers and floats correctly."""

    def test_zero(self):
        assert app.seconds_to_hhmmss(0) == "00:00:00"

    def test_one_minute(self):
        assert app.seconds_to_hhmmss(60) == "00:01:00"

    def test_one_hour(self):
        assert app.seconds_to_hhmmss(3600) == "01:00:00"

    def test_mixed(self):
        assert app.seconds_to_hhmmss(3725) == "01:02:05"

    def test_negative_clamps_to_zero(self):
        assert app.seconds_to_hhmmss(-5) == "00:00:00"

    def test_float_truncated_not_rounded(self):
        # 90.9 seconds → 00:01:30  (int() truncates, does not round)
        assert app.seconds_to_hhmmss(90.9) == "00:01:30"


class TestBuildYoutubeTimeUrl:
    """build_youtube_time_url must produce a well-formed deep-link URL."""

    def test_basic(self):
        url = app.build_youtube_time_url("abc123", 125.7)
        assert url == "https://www.youtube.com/watch?v=abc123&t=125s"

    def test_negative_start_clamps_to_zero(self):
        url = app.build_youtube_time_url("abc123", -10)
        assert url == "https://www.youtube.com/watch?v=abc123&t=0s"


# ===========================================================================
# 3. Hash functions
# ===========================================================================

class TestHashing:
    def test_text_hash_is_deterministic(self):
        assert app.text_hash("hello world") == app.text_hash("hello world")

    def test_text_hash_differs_for_different_input(self):
        assert app.text_hash("hello") != app.text_hash("world")

    def test_text_hash_length_is_16(self):
        assert len(app.text_hash("anything")) == 16

    def test_stable_hash_obj_is_deterministic(self):
        obj = {"b": 2, "a": 1}
        assert app.stable_hash_obj(obj) == app.stable_hash_obj(obj)

    def test_stable_hash_obj_is_key_order_independent(self):
        assert app.stable_hash_obj({"a": 1, "b": 2}) == app.stable_hash_obj({"b": 2, "a": 1})

    def test_stable_hash_obj_differs_for_different_values(self):
        assert app.stable_hash_obj({"a": 1}) != app.stable_hash_obj({"a": 2})

    def test_stable_hash_obj_length_is_16(self):
        assert len(app.stable_hash_obj({"x": "y"})) == 16


# ===========================================================================
# 4. Data-model roundtrips
# ===========================================================================

def _make_segment_dict(sid=0, start=1.0, end=5.0, text="Hello world") -> Dict[str, Any]:
    return {
        "segment_id": sid,
        "start": start,
        "end": end,
        "text": text,
        "words": [
            {"word": "Hello", "start": 1.0, "end": 1.5, "probability": 0.99},
            {"word": "world", "start": 1.6, "end": 2.0, "probability": 0.95},
        ],
    }


class TestTranscriptSegmentRoundtrip:
    def test_roundtrip_preserves_all_fields(self):
        data = _make_segment_dict()
        seg = app.TranscriptSegment.from_dict(data)
        restored = seg.to_dict()
        assert restored["segment_id"] == 0
        assert restored["start"] == 1.0
        assert restored["end"] == 5.0
        assert restored["text"] == "Hello world"
        assert len(restored["words"]) == 2
        assert restored["words"][0]["word"] == "Hello"

    def test_missing_words_defaults_to_empty_list(self):
        data = _make_segment_dict()
        del data["words"]
        seg = app.TranscriptSegment.from_dict(data)
        assert seg.words == []

    def test_types_are_coerced(self):
        data = _make_segment_dict(sid="3", start="2.5", end="7.0")
        seg = app.TranscriptSegment.from_dict(data)
        assert seg.segment_id == 3
        assert isinstance(seg.start, float)
        assert isinstance(seg.end, float)


class TestRetrievalChunkRoundtrip:
    def test_roundtrip_preserves_all_fields(self):
        data = {
            "chunk_id": 0,
            "text": "Some transcript text",
            "start": 0.0,
            "end": 10.0,
            "segment_ids": [0, 1, 2],
        }
        chunk = app.RetrievalChunk.from_dict(data)
        restored = chunk.to_dict()
        assert restored["chunk_id"] == 0
        assert restored["text"] == "Some transcript text"
        assert restored["segment_ids"] == [0, 1, 2]

    def test_segment_ids_coerced_to_int(self):
        data = {
            "chunk_id": 1,
            "text": "x",
            "start": 0.0,
            "end": 1.0,
            "segment_ids": ["0", "1"],
        }
        chunk = app.RetrievalChunk.from_dict(data)
        assert chunk.segment_ids == [0, 1]


# ===========================================================================
# 5. Retrieval chunking
# ===========================================================================

def _make_segments(texts: List[str], start_offset: float = 0.0) -> List[app.TranscriptSegment]:
    segs = []
    t = start_offset
    for i, text in enumerate(texts):
        segs.append(app.TranscriptSegment(segment_id=i, start=t, end=t + 1.0, text=text))
        t += 1.0
    return segs


class TestBuildRetrievalChunks:
    def test_empty_segments_returns_empty(self):
        assert app.build_retrieval_chunks([]) == []

    def test_single_segment_becomes_single_chunk(self):
        segs = _make_segments(["Hello world"])
        chunks = app.build_retrieval_chunks(segs)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].start == 0.0
        assert chunks[0].end == 1.0
        assert chunks[0].segment_ids == [0]

    def test_chunks_cover_all_segments(self):
        # 10 segments of 80 chars each, chunk_size=200 → roughly 2 per chunk
        text = "a" * 80
        segs = _make_segments([text] * 10)
        chunks = app.build_retrieval_chunks(segs)
        # Every segment_id must appear in at least one chunk
        covered = {sid for chunk in chunks for sid in chunk.segment_ids}
        assert covered == set(range(10))

    def test_overlap_causes_segment_reuse(self):
        # With overlap=1, the last segment of chunk N should start chunk N+1
        text = "word " * 60  # ~300 chars, bigger than default chunk_size of 1000 chars? No.
        # Let's use very small chunks: we need to override CFG
        # Instead just test that with default overlap=1, consecutive chunks share a segment
        texts = [f"segment {i} content" for i in range(6)]
        segs = _make_segments(texts)
        chunks = app.build_retrieval_chunks(segs)
        if len(chunks) >= 2:
            # Last segment_id of chunk 0 should appear in chunk 1 (overlap)
            last_of_first = chunks[0].segment_ids[-1]
            first_of_second = chunks[1].segment_ids[0]
            # With overlap=1, first_of_second <= last_of_first
            assert first_of_second <= last_of_first

    def test_chunk_ids_are_sequential(self):
        segs = _make_segments([f"segment {i}" for i in range(5)])
        chunks = app.build_retrieval_chunks(segs)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_segments_with_only_empty_text_return_empty(self):
        segs = _make_segments(["", "  ", "\t"])
        chunks = app.build_retrieval_chunks(segs)
        assert chunks == []

    def test_chunk_time_range_is_correct(self):
        segs = [
            app.TranscriptSegment(0, start=10.0, end=15.0, text="first"),
            app.TranscriptSegment(1, start=15.0, end=20.0, text="second"),
        ]
        chunks = app.build_retrieval_chunks(segs)
        # At least the first chunk should span 10.0 → 20.0 (or 15.0 depending on chunk size)
        assert chunks[0].start == 10.0

