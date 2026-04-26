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

