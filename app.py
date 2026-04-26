"""
[WIP] YouTube STT Summarizer & Timestamp-Aware Q&A Tool

Business goal:
- Download audio from a YouTube video.
- Transcribe it locally with faster-whisper.
- Cache transcript and retrieval artifacts so repeated runs stay fast.
- Summarize the transcript with Ollama.
- Answer questions from the transcript with FAISS retrieval.
- Render clickable YouTube timestamp links in Q&A answers.
"""




import logging
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AppConfig:
    """Application configuration and cache-version inputs.

    Business logic:
    cache correctness depends on runtime settings. When model or chunking
    settings change, cache keys should change too.

    Note: embed_chunk_size is measured in characters, not tokens.
    """

    base_dir: Path = Path(__file__).resolve().parent
    llm_model: str = "llama3.1:8b-instruct-q8_0"
    embedding_model: str = "mxbai-embed-large" # "nomic-embed-text" or "mxbai-embed-large"
    ollama_base_url: str = "http://localhost:11434"

    llm_context_limit: int = 8192
    safety_margin: int = 1024

    summary_chunk_overlap_tokens: int = 256
    max_summary_passes: int = 4

    # Character-based limit for retrieval chunks (not token-based).
    embed_chunk_size: int = 1000
    embed_chunk_overlap_segments: int = 1
    retrieval_top_k: int = 4

    whisper_model_size: str = "small"   # "small", "Medium", "large-v3"
    whisper_device: str = "cpu"         # "cpu" or "cuda"
    whisper_compute_type: str = "int8"  # "int8" or "float16"
    whisper_language: Optional[str] = None
    whisper_beam_size: int = 1          # usually 1-5 
    whisper_vad_filter: bool = False    # quality: True, speed: False
    whisper_condition_on_previous_text: bool = False    # quality: True, speed: False
    whisper_word_timestamps: bool = True

    summary_prompt_version: str = "summary-v1"
    retrieval_prompt_version: str = "qa-with-timestamps-v1"
    transcript_schema_version: str = "timestamped-transcript-v1"
    retrieval_schema_version: str = "timestamped-retrieval-v1"
    trust_faiss_cache: bool = True

    @property
    def max_transcript_tokens(self) -> int:
        return self.llm_context_limit - self.safety_margin

    @property
    def summary_target_tokens(self) -> int:
        return self.max_transcript_tokens // 2
    

@dataclass(frozen=True)
class CachePaths:
    """Resolved on-disk cache locations for the application."""

    root: Path
    ytdlp: Path
    audio: Path
    transcript: Path
    summary: Path
    retrieval: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "CachePaths":
        cache_root = base_dir / "cache"
        return cls(
            root=cache_root,
            ytdlp=cache_root / "yt_dlp_cache",
            audio=cache_root / "audio_cache",
            transcript=cache_root / "transcript_cache",
            summary=cache_root / "summary_cache",
            retrieval=cache_root / "retrieval_cache",
        )

    def ensure(self) -> None:
        for path in (self.root, self.ytdlp, self.audio, self.transcript, self.summary, self.retrieval):
            path.mkdir(parents=True, exist_ok=True)


CFG = AppConfig()
PATHS = CachePaths.from_base_dir(CFG.base_dir)

# =============================================================================
# GENERAL UTILS
# =============================================================================

def run_command(cmd: List[str]) -> None:
    """Execute a subprocess command and raise a readable error if it fails."""
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    

def remove_matching_files(directory: Path, prefix: str) -> None:
    """Delete stale files that share the same prefix inside a cache folder."""
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.name.startswith(prefix):
            try:
                path.unlink()
                logger.info("Stale files deleted: %s", path)
            except FileNotFoundError:
                pass


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert floating-point seconds to HH:MM:SS."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def build_youtube_time_url(video_id: str, start_seconds: float) -> str:
    """Build a YouTube watch URL with a stable start timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={max(0, int(start_seconds))}s"


@lru_cache(maxsize=128)
def get_video_id(url: str) -> Optional[str]:
    """Extract a YouTube video id from common URL shapes."""
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None 


def require_video_id(video_url: str) -> str:
    """Return video id or raise a validation error."""
    video_id = get_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")
    logger.info("Video id found: %s", video_id)
    return video_id


def download_audio(video_url: str, output_dir: Path) -> Path:
    """Download source audio using yt-dlp.

    Business logic:
    keep the source download separate from ffmpeg normalization so failures are
    easier to reason about and retry.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_matching_files(output_dir, "source.")
    out_template = str(output_dir / "source.%(ext)s")
    run_command(["yt-dlp", "-f", "bestaudio/best", "-o", out_template, video_url])

    candidates = sorted(output_dir.glob("source.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("yt-dlp did not produce an audio file.")
    return candidates[-1]


def convert_to_wav_16k_mono(input_audio: Path, output_wav: Path) -> Path:
    """Normalize audio into 16kHz mono WAV for local Whisper inference."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_audio),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_wav),
        ]
    )
    return output_wav


video_url = "https://www.youtube.com/watch?v=BSuAgw8Lc1Y"
video_id = require_video_id(video_url)

source_dir = PATHS.ytdlp / video_id
audio_dir = PATHS.audio / video_id
wav_path = audio_dir / "audio.wav"

source_audio = download_audio(video_id, source_dir)
convert_to_wav_16k_mono(source_audio, wav_path)
logger.info(build_youtube_time_url(video_id, 245))

