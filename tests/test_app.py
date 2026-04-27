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

