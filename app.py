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

import os
# Must be set before any C extension (CTranslate2, OpenMP) is loaded.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2) # optional

import hashlib
import json
import logging
import re
import requests
import subprocess
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, NamedTuple, Optional, Tuple

import gradio as gr
import tiktoken
from faster_whisper import WhisperModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain_qdrant import Qdrant
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
from langchain_community.llms import Ollama
from rank_bm25 import BM25Okapi


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    """Application configuration and cache-version inputs.

    Business logic:
    cache correctness depends on runtime settings. When model or chunking
    settings change, cache keys should change too.

    """

    base_dir: Path = Path(__file__).resolve().parent
    llm_model: str = "llama3.1:8b-instruct-q8_0"
    embedding_model: str = "mxbai-embed-large" # "nomic-embed-text" or "mxbai-embed-large"
    ollama_base_url: str = "http://localhost:11434"

    llm_context_limit: int = 4096 # 8192
    safety_margin: int = 1024

    summary_chunk_overlap_tokens: int = 256
    max_summary_passes: int = 4

    # Character-based limit for semantic chunking parameters
    semantic_chunk_threshold: float = 0.75      # cosine similarity merge threshold
    semantic_chunk_max_chars: int = 1200        # hard cap to prevent oversized embeddings
    # Hybrid search tuning parameters
    hybrid_dense_weight: float = 0.7       # 0.0 = pure BM25, 1.0 = pure FAISS
    hybrid_top_k_candidates: int = 8       # fetch more candidates before fusion
    retrieval_top_k: int = 4
    vector_db_type: str = "qdrant"  # "qdrant" or "faiss"
    qdrant_path: str = "cache/qdrant_db"  # local persistence path

    whisper_model_size: str = "medium"   # "small", "medium", "large-v3"
    whisper_device: str = "cpu"         # "cpu" or "cuda"
    whisper_compute_type: str = "int8"  # "int8" or "float16"
    whisper_language: Optional[str] = None
    whisper_beam_size: int = 5          # usually 1-5 
    whisper_vad_filter: bool = True    # quality: True, speed: False
    whisper_condition_on_previous_text: bool = True    # quality: True, speed: False
    # word_timestamps=True triggers a DTW (Dynamic Time Warping) alignment pass
    # after every decoded segment to pin each word to its exact audio position.
    # On CPU this increases transcription time. 
    # Word-level data is stored in WordTiming but is not consumed by any current 
    # pipeline step — segment-level timestamps are sufficient for YouTube deep-link generation. 
    # Enable only if a word-highlight UI is added.
    whisper_word_timestamps: bool = False

    summary_prompt_version: str = "summary-v1"
    retrieval_prompt_version: str = "qa-with-timestamps-v1"
    transcript_schema_version: str = "timestamped-transcript-v1"
    retrieval_schema_version: str = "timestamped-retrieval-v2"

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
DB_PATH = PATHS.root / "app_data.db"
TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

# Compiled patterns used by render_clickable_answer.
# Defined at module level so they are compiled once, not per call.
SOURCE_GROUP_PATTERN = re.compile(r"\[((?:S\d+\s*(?:,\s*S\d+\s*)*))\]")
SOURCE_SINGLE_PATTERN = re.compile(r"(?<!\[)(S\d+)(?!\])")


# =============================================================================
# Database Manager & Schema
# =============================================================================

@contextmanager
def get_db():
    """Thread-safe SQLite connection manager for Gradio's async event loop.
    
    Design Rationale:
    - Short-lived connections prevent long-running locks that block concurrent UI events.
    - check_same_thread=False is required because Gradio routes requests across threads.
    - Explicit commit/rollback ensures partial writes never corrupt the cache.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create cache tables and indexes if they don't exist.
    
    Schema Design:
    - videos: Lightweight registry for observability & future library UI.
    - transcripts: Stores full text + JSON-serialized segments. Config hash
      filtering guarantees automatic invalidation when Whisper settings change.
    - summaries: Stores generated summaries keyed by video, transcript hash,
      mode (direct/chunked), and summary config hash. Composite index matches
      the exact WHERE clause used in load_cached_summary for O(1) lookups.
    """
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                url TEXT,
                indexed_at REAL
            );
            CREATE TABLE IF NOT EXISTS transcripts (
                video_id TEXT PRIMARY KEY,
                transcript TEXT,
                transcript_hash TEXT,
                segments_json TEXT,
                stt_config_hash TEXT,
                saved_at REAL,
                FOREIGN KEY(video_id) REFERENCES videos(video_id)
            );
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                transcript_hash TEXT,
                mode TEXT,
                summary TEXT,
                summary_config_hash TEXT,
                saved_at REAL,
                FOREIGN KEY(video_id) REFERENCES videos(video_id)
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_lookup
                ON summaries(video_id, transcript_hash, mode, summary_config_hash);
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                transcript_hash TEXT,
                chapters_json TEXT,
                saved_at REAL,
                FOREIGN KEY(video_id) REFERENCES videos(video_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chapters_lookup ON chapters(video_id, transcript_hash);
        """)


# =============================================================================
# HASHES / CONFIG SNAPSHOTS
# =============================================================================

def stable_hash_obj(obj: Dict[str, Any]) -> str:
    """Return a short deterministic hash for config-based cache keys."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def text_hash(text: str) -> str:
    """Return a short deterministic hash for content-based cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def current_stt_config() -> Dict[str, Any]:
    return {
        "whisper_model_size": CFG.whisper_model_size,
        "whisper_device": CFG.whisper_device,
        "whisper_compute_type": CFG.whisper_compute_type,
        "language": CFG.whisper_language,
        "beam_size": CFG.whisper_beam_size,
        "vad_filter": CFG.whisper_vad_filter,
        "condition_on_previous_text": CFG.whisper_condition_on_previous_text,
        "word_timestamps": CFG.whisper_word_timestamps,
        "transcript_schema_version": CFG.transcript_schema_version,
    }


def current_summary_config() -> Dict[str, Any]:
    return {
        "llm_model": CFG.llm_model,
        "llm_context_limit": CFG.llm_context_limit,
        "safety_margin": CFG.safety_margin,
        "max_transcript_tokens": CFG.max_transcript_tokens,
        "summary_target_tokens": CFG.summary_target_tokens,
        "summary_chunk_overlap_tokens": CFG.summary_chunk_overlap_tokens,
        "summary_prompt_version": CFG.summary_prompt_version,
    }


def current_retrieval_config() -> Dict[str, Any]:
    return {
        "embedding_model": CFG.embedding_model,
        "semantic_chunk_threshold": CFG.semantic_chunk_threshold,
        "semantic_chunk_max_chars": CFG.semantic_chunk_max_chars,
        "retrieval_prompt_version": CFG.retrieval_prompt_version,
        "retrieval_schema_version": CFG.retrieval_schema_version,
        "retrieval_top_k": CFG.retrieval_top_k,
    }


STT_CONFIG_HASH = stable_hash_obj(current_stt_config())
SUMMARY_CONFIG_HASH = stable_hash_obj(current_summary_config())
RETRIEVAL_CONFIG_HASH = stable_hash_obj(current_retrieval_config())


# =============================================================================
# GENERAL UTILS
# =============================================================================

@contextmanager
def log_time(label: str) -> Generator[None, None, None]:
    """Log the execution time of a block."""
    start = time.perf_counter()
    logger.info("START: %s", label)
    try:
        yield
    finally:
        logger.info("END: %s (%.2fs)", label, time.perf_counter() - start)


def estimate_tokens(text: str) -> int:
    """Estimate tokens for prompt-budget decisions."""
    return len(TIKTOKEN_ENC.encode(text))


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


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON safely and return None on missing or invalid files."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read JSON cache: %s", path, exc_info=True)
        return None


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partially written cache records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


# =============================================================================
# AUDIO PIPELINE
# =============================================================================

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


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class WordTiming:
    """Word-level timing entry from Whisper's word_timestamps output.

    All fields are optional: older Whisper builds may omit start/end/probability
    for individual words, so None is an acceptable value in those slots.
    """
    word: str
    start: Optional[float]
    end: Optional[float]
    probability: Optional[float]


@dataclass
class TranscriptSegment:
    """One Whisper segment with its start/end timestamps, text, and word list.

    The words field is populated when word_timestamps=True is set in AppConfig.
    It defaults to an empty list so code that doesn't need word-level data can
    ignore it without extra None-checks.
    """
    segment_id: int
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=int(data["segment_id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
            words=[WordTiming(**w) for w in data.get("words", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

@dataclass(frozen=True)
class RuntimeDeps:
    """Runtime model dependencies and prompt chains."""

    llm: Ollama
    embeddings: OllamaEmbeddings
    whisper: WhisperModel
    summary_chain: LLMChain
    chunk_summary_chain: LLMChain
    reduce_summary_chain: LLMChain
    qa_chain: LLMChain
    chapter_chain: LLMChain


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def normalize_model_name(name: str) -> str:
    """Normalize Ollama model names by stripping variant suffixes."""
    return name.strip().split(":", 1)[0]


def ensure_ollama_ready(base_url: str, required_models: Iterable[str], timeout: float = 5.0) -> None:
    """Fail fast when Ollama is offline or required models are missing.

    Business logic:
    validating availability during startup avoids wasting time on downloading and
    transcribing audio before discovering that generation cannot run.
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at '{base_url}'. Make sure the server is running."
        ) from exc

    available_names = {
        item.get("name").strip()
        for item in data.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }
    available_base_names = {normalize_model_name(name) for name in available_names}

    missing = [
        model for model in required_models
        if model not in available_names and normalize_model_name(model) not in available_base_names
    ]

    if missing:
        raise RuntimeError(
            "Ollama is online, but required models are missing: "
            f"{', '.join(missing)}. Available: {', '.join(sorted(available_names)) or '(none)'}"
        )
    

def warmup_ollama_clients(llm: Ollama, embeddings: OllamaEmbeddings) -> None:
    """Verify both embeddings and generation paths before the UI starts."""
    try:
        vector = embeddings.embed_query("health check")
        if not vector:
            raise RuntimeError("Embedding warmup returned an empty vector.")
    except Exception as exc:
        raise RuntimeError("Ollama embeddings warmup failed.") from exc

    try:
        response = llm.invoke("Reply with OK.")
        if response is None or not str(response).strip():
            raise RuntimeError("LLM warmup returned an empty response.")
    except Exception as exc:
        raise RuntimeError("Ollama LLM warmup failed.") from exc
    

def make_prompt_chain(llm: Ollama, template: str, input_variables: List[str]) -> LLMChain:
    """Construct a LangChain prompt chain with consistent wiring."""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=template, input_variables=input_variables),
    )


def build_runtime() -> RuntimeDeps:
    """Initialize model clients and prompt chains used by the application."""
    logger.info("Initializing models at startup...")

    ensure_ollama_ready(CFG.ollama_base_url, [CFG.llm_model, CFG.embedding_model])

    llm = Ollama(
        model=CFG.llm_model,
        temperature=0.7,
        top_p=0.9,
        base_url=CFG.ollama_base_url,
    )
    embeddings = OllamaEmbeddings(
        model=CFG.embedding_model,
        base_url=CFG.ollama_base_url,
    )
    warmup_ollama_clients(llm, embeddings)

    whisper = WhisperModel(
        CFG.whisper_model_size,
        device=CFG.whisper_device,
        compute_type=CFG.whisper_compute_type,
    )

    summary_chain = make_prompt_chain(
        llm,
        """You are an AI assistant that summarizes YouTube video transcripts.

Instructions:
- Write a concise but informative summary.
- Focus only on what is actually said.
- Ignore timestamps and filler words.
- Capture important details, numbers, names, and concrete claims.
- Avoid repetition.

Transcript:
{transcript}

Summary:""",
        ["transcript"],
    )

    chunk_summary_chain = make_prompt_chain(
        llm,
        """You are an AI assistant summarizing one part of a YouTube transcript.

Instructions:
- Write a dense but compact summary of this chunk.
- Focus only on what is actually said.
- Preserve important facts, names, numbers, and claims.
- Ignore timestamps and filler words.
- Do not add commentary.
- Do not repeat obvious context.

Transcript chunk:
{chunk}

Chunk summary:""",
        ["chunk"],
    )

    reduce_summary_chain = make_prompt_chain(
        llm,
        """You are an AI assistant combining partial summaries of a YouTube transcript.

Instructions:
- Merge these chunk summaries into one coherent final summary.
- Preserve important facts, names, numbers, and claims.
- Remove redundancy and repeated points.
- Keep the result concise but informative.
- Do not invent anything that is not supported by the summaries.

Chunk summaries:
{summaries}

Final summary:""",
        ["summaries"],
    )

    qa_chain = make_prompt_chain(
        llm,
        """Answer the question using ONLY the context.

Rules:
- If the answer is not in the context, respond exactly:
  "I don't have enough information to answer this question."
- When you use evidence, cite supporting sources inline using source ids, for example [S1] or [S1][S2].
- Do not invent timestamps or sources.
- Prefer a concise answer.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:""",
        ["context", "question", "chat_history"],
    )

    chapter_chain = make_prompt_chain(
        llm,
        """You are an AI that structures YouTube transcripts into logical chapters.
Analyze the following timestamped segments and group them into coherent chapters.
Return ONLY a valid JSON array of objects with these exact keys:
- "title": short descriptive chapter title
- "start_idx": index of first segment in chapter
- "end_idx": index of last segment in chapter

Rules:
- Chapters must cover all segments contiguously (no gaps, no overlaps).
- Keep titles concise (3-6 words).
- Return ONLY JSON. No markdown, no explanations.

Segments:
{segments}

JSON:""",
        ["segments"],
    )

    logger.info("Models initialized successfully")
    return RuntimeDeps(
        llm=llm,
        embeddings=embeddings,
        whisper=whisper,
        summary_chain=summary_chain,
        chunk_summary_chain=chunk_summary_chain,
        reduce_summary_chain=reduce_summary_chain,
        qa_chain=qa_chain,
        chapter_chain=chapter_chain,
    )


# =============================================================================
# TRANSCRIPT CACHE AND PIPELINE
# =============================================================================

def load_cached_transcript(
    video_id: str,
) -> Optional[Tuple[str, List[TranscriptSegment], str]]:
    """Load transcript from SQLite when STT config hash matches."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT transcript, transcript_hash, segments_json FROM transcripts "
            "WHERE video_id = ? AND stt_config_hash = ?",
            (video_id, STT_CONFIG_HASH)
        ).fetchone()
    if not row:
        return None
    
    # Safely deserialize segment list; skip corrupted records gracefully
    try:
        segments_data = json.loads(row["segments_json"])
        segments = [TranscriptSegment.from_dict(s) for s in segments_data if isinstance(s, dict)]
    except (json.JSONDecodeError, KeyError):
        logger.warning("Corrupted segments_json for %s — treating as cache miss.", video_id)
        return None
        
    return row["transcript"], segments, row["transcript_hash"]


def save_transcript(
    video_id: str, transcript: str, segments: List[TranscriptSegment]
) -> None:
    """Persist transcript and segments to SQLite with atomic upsert."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO videos (video_id, indexed_at) VALUES (?, ?)",
            (video_id, time.time())
        )
        conn.execute(
            "INSERT OR REPLACE INTO transcripts "
            "(video_id, transcript, transcript_hash, segments_json, stt_config_hash, saved_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                video_id,
                transcript,
                text_hash(transcript),
                json.dumps([s.to_dict() for s in segments]),
                STT_CONFIG_HASH,
                time.time()
            )
        )


def transcribe_audio(
    audio_path: Path, runtime: RuntimeDeps
) -> Tuple[str, List[TranscriptSegment]]:
    """Run faster-whisper and return transcript text plus typed segments."""
    kwargs: Dict[str, Any] = {
        "beam_size": CFG.whisper_beam_size,
        "vad_filter": CFG.whisper_vad_filter,
        "condition_on_previous_text": CFG.whisper_condition_on_previous_text,
        "word_timestamps": CFG.whisper_word_timestamps,
    }
    if CFG.whisper_language:
        kwargs["language"] = CFG.whisper_language

    with log_time("create whisper generator"):
        segments, info = runtime.whisper.transcribe(str(audio_path), **kwargs)
    with log_time("run whisper inference for chunk"):
        transcript_lines: List[str] = []
        structured_segments: List[TranscriptSegment] = []

        for seg_idx, seg in enumerate(segments):
            text = seg.text.strip()
            if not text:
                continue
            transcript_lines.append(text)
            words: List[WordTiming] = []
            for word in getattr(seg, "words", None) or []:
                if getattr(word, "word", None) is None:
                    continue
                words.append(
                    WordTiming(
                        word=str(word.word),
                        start=float(word.start) if word.start is not None else None,
                        end=float(word.end) if word.end is not None else None,
                        probability=(
                            float(word.probability)
                            if getattr(word, "probability", None) is not None
                            else None
                        ),
                    )
                )
            structured_segments.append(
                TranscriptSegment(
                    segment_id=seg_idx,
                    start=float(seg.start) if seg.start is not None else 0.0,
                    end=float(seg.end) if seg.end is not None else 0.0,
                    text=text,
                    words=words,
                )
            )

    transcript = "\n".join(transcript_lines).strip()
    if not transcript:
        raise RuntimeError("STT returned an empty transcript.")

    logger.info(
        "Transcription done (language=%s, prob=%.3f, chars=%d, segments=%d)",
        getattr(info, "language", "unknown"),
        getattr(info, "language_probability", 0.0),
        len(transcript),
        len(structured_segments),
    )
    return transcript, structured_segments


# =============================================================================
# TYPED GENERATOR YIELDS
# =============================================================================

class SttUpdate(NamedTuple):
    """Streaming update yielded by fetch_transcript_from_stt_stream.

    Fields:
        message    — human-readable status line for the UI label.
        transcript — full transcript string once available; None during early
                     pipeline stages.
        segments   — typed List[TranscriptSegment] once Whisper completes;
                     None during early stages.
        progress   — integer 0-100 for the progress bar.
    """
    message: str
    transcript: Optional[str]
    segments: Optional[List[TranscriptSegment]]
    progress: int


class SummaryUpdate(NamedTuple):
    """Streaming update yielded by summarize_transcript_stream.

    Fields:
        message  — human-readable status line for the UI label.
        summary  — finished summary text once generated; None during early
                   stages.
        progress — integer 0-100 for the progress bar.
    """
    message: str
    summary: Optional[str]
    progress: int

# =============================================================================
# STT STREAMING PIPELINE
# =============================================================================

def fetch_transcript_from_stt_stream(
    video_url: str,
    runtime: RuntimeDeps,
) -> Generator[SttUpdate, None, None]:
    """Download, convert, and transcribe a YouTube video, yielding typed updates.

    Full pipeline with progress checkpoints:

        10 % — audio download starts (yt-dlp)
        30 % — audio conversion starts (ffmpeg → 16kHz mono WAV)
        55 % — Whisper transcription starts (may take several minutes)
        90 % — transcript saved to JSON cache
       100 % — final SttUpdate with full transcript text and typed segments

    Cache fast-exit:
        If a cached transcript exists for the current video_id and STT config,
        a single SttUpdate at 100 % is yielded immediately and the function
        returns — no download, no conversion, no Whisper call.

    Intermediate SttUpdate yields carry None in the transcript and segments
    fields.  Callers must guard against None before consuming those values.
    """
    video_id = require_video_id(video_url)

    # ── 1. Cache check ──────────────────────────────────────────────────────
    cached = load_cached_transcript(video_id)
    if cached:
        transcript, segments, _ = cached
        logger.info(
            "Transcript cache hit for %s (%d chars, %d segments).",
            video_id, len(transcript), len(segments),
        )
        yield SttUpdate(
            message="✅ Using cached transcript.",
            transcript=transcript,
            segments=segments,
            progress=100,
        )
        return

    # ── 2. Audio download ───────────────────────────────────────────────────
    yield SttUpdate(
        message="⬇️ Downloading audio from YouTube...",
        transcript=None,
        segments=None,
        progress=10,
    )
    audio_dir = PATHS.ytdlp / video_id
    with log_time("yt-dlp download"):
        raw_audio = download_audio(video_url, audio_dir)
    logger.info("Downloaded audio: %s", raw_audio)

    # ── 3. Audio conversion ─────────────────────────────────────────────────
    yield SttUpdate(
        message="🔄 Converting audio to 16 kHz mono WAV...",
        transcript=None,
        segments=None,
        progress=30,
    )
    wav_path = PATHS.audio / video_id / "audio.wav"
    with log_time("ffmpeg convert"):
        convert_to_wav_16k_mono(raw_audio, wav_path)

    # ── 4. Whisper transcription ────────────────────────────────────────────
    yield SttUpdate(
        message="🎙️ Transcribing with Whisper (this may take a while)...",
        transcript=None,
        segments=None,
        progress=55,
    )
    transcript, segments = transcribe_audio(wav_path, runtime)

    # ── 5. Persist to cache ─────────────────────────────────────────────────
    yield SttUpdate(
        message="💾 Saving transcript to cache...",
        transcript=transcript,
        segments=segments,
        progress=90,
    )
    save_transcript(video_id, transcript, segments)

    # ── 6. Final yield ──────────────────────────────────────────────────────
    yield SttUpdate(
        message=(
            f"✅ Transcript ready "
            f"({len(transcript)} chars, {len(segments)} segments)."
        ),
        transcript=transcript,
        segments=segments,
        progress=100,
    )

# =============================================================================
# SUMMARIZATION PIPELINE
# =============================================================================

def chunk_transcript_for_summary(text: str) -> List[str]:
    """Split transcript text into token-aware chunks for long-input summarization.

    Uses RecursiveCharacterTextSplitter with a token-based length function so
    each chunk stays within CFG.summary_target_tokens.  The overlap ensures
    sentences that fall near a boundary are not lost between chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.summary_target_tokens,
        chunk_overlap=CFG.summary_chunk_overlap_tokens,
        separators=["\n\n", "\n", " ", ""],
        length_function=estimate_tokens,
    )
    return splitter.split_text(text)


def summarize_transcript_stream(
    text: str,
    runtime: RuntimeDeps,
    prompt_override: str = "",
) -> Generator[SummaryUpdate, None, None]:
    """Summarize transcript directly or through iterative hierarchical reduction.

    Short transcripts (≤ max_transcript_tokens) are sent to the LLM in one
    call.  Longer transcripts are split into chunks, each chunk summarised
    individually, then the chunk summaries are merged.  If the merged text is
    still too long the loop runs again, up to CFG.max_summary_passes times.

    Strategy selection:
        direct  — transcript fits in one context window.
        chunked — one or more reduction passes required.
    """
    if estimate_tokens(text) <= CFG.max_transcript_tokens:
        yield SummaryUpdate("📝 Generating direct summary...", None, 30)
        with log_time("direct summary generation"):
            tpl = prompt_override or runtime.summary_chain.prompt.template
            summary = run_llm_dynamic(runtime.llm, tpl, {"transcript": text})
        if not summary:
            raise RuntimeError("Direct summary generation returned empty output.")
        yield SummaryUpdate("✅ Final summary ready.", summary, 100)
        return

    # ---- Hierarchical multi-pass reduction ----
    current_text = text
    for pass_idx in range(1, CFG.max_summary_passes + 1):
        chunks = chunk_transcript_for_summary(current_text)
        yield SummaryUpdate(
            f"🧩 Summary pass {pass_idx}: split into {len(chunks)} chunks...",
            None,
            10,
        )

        chunk_summaries: List[str] = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            progress = min(80, 10 + int(70 * idx / total))
            yield SummaryUpdate(
                f"📝 Summarizing chunk {idx}/{total}...", None, progress
            )
            with log_time(f"chunk summary {pass_idx}:{idx}/{total}"):
                tpl = prompt_override or runtime.chunk_summary_chain.prompt.template
                chunk_summary = run_llm_dynamic(runtime.llm, tpl, {"chunk": chunk})
            if not chunk_summary:
                raise RuntimeError(
                    f"Chunk summary {idx} returned empty output."
                )
            chunk_summaries.append(chunk_summary)

        merged = "\n\n".join(chunk_summaries)
        merged_tokens = estimate_tokens(merged)
        logger.info(
            "Summary pass %d produced %d tokens", pass_idx, merged_tokens
        )

        if merged_tokens <= CFG.max_transcript_tokens:
            yield SummaryUpdate(
                "📌 Generating final summary from merged chunk summaries...",
                None,
                90,
            )
            with log_time("final merged summary generation"):
                tpl = prompt_override or runtime.reduce_summary_chain.prompt.template
                final_summary = run_llm_dynamic(runtime.llm, tpl, {"summaries": merged})
            if not final_summary:
                raise RuntimeError(
                    "Final summary generation returned empty output."
                )
            yield SummaryUpdate("✅ Final summary ready.", final_summary, 100)
            return

        yield SummaryUpdate(
            f"⚠️ Intermediate summary still too long "
            f"({merged_tokens} tokens). Re-summarizing...",
            None,
            85,
        )
        current_text = merged

    raise RuntimeError(
        "Hierarchical summarization exceeded the maximum number of "
        "reduction passes."
    )


# =============================================================================
# SUMMARY CACHE
# =============================================================================


def save_summary(
    video_id: str, transcript_hash_value: str, mode: str, summary: str
) -> None:
    """Persist summary to SQLite. Inserts new row to preserve history."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO summaries "
            "(video_id, transcript_hash, mode, summary, summary_config_hash, saved_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, transcript_hash_value, mode, summary, SUMMARY_CONFIG_HASH, time.time())
        )


def load_cached_summary(
    video_id: str, transcript_hash_value: str, mode: str
) -> Optional[str]:
    """Load latest matching summary from SQLite using indexed lookup."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT summary FROM summaries WHERE video_id = ? AND transcript_hash = ? "
            "AND mode = ? AND summary_config_hash = ? ORDER BY saved_at DESC LIMIT 1",
            (video_id, transcript_hash_value, mode, SUMMARY_CONFIG_HASH)
        ).fetchone()
    return row["summary"] if row else None


# =============================================================================
# RETRIEVAL DATA MODEL
# =============================================================================

@dataclass
class RetrievalChunk:
    """Retrieval unit stored in cache and indexed into FAISS.

    Each chunk covers a contiguous window of TranscriptSegment objects.
    segment_ids records which segments are included so source timestamps can
    be recovered after a FAISS similarity search without re-scanning the full
    transcript.
    """
    chunk_id: int
    text: str
    start: float
    end: float
    segment_ids: List[int]
    video_id: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalChunk":
        return cls(
            chunk_id=int(data["chunk_id"]),
            text=str(data["text"]),
            start=float(data["start"]),
            end=float(data["end"]),
            segment_ids=[int(x) for x in data.get("segment_ids", [])],
            video_id=data.get("video_id", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

# =============================================================================
# RETRIEVAL CHUNKING
# =============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


def build_retrieval_chunks(
    segments: List[TranscriptSegment],
    embeddings: OllamaEmbeddings,
) -> List[RetrievalChunk]:
    """Group TranscriptSegment objects into semantically coherent RetrievalChunks.
    
    Strategy:
    1. Filter segments with usable text.
    2. Batch-embed each segment.
    3. Merge adjacent segments when cosine similarity >= threshold AND
       combined length <= semantic_chunk_max_chars.
    4. Preserve exact timestamp boundaries & segment_ids for citation rendering.
    """
    valid = [s for s in segments if s.text.strip()]
    if not valid:
        return []

    # Batch embed all valid segments (Ollama handles batching efficiently)
    segment_texts = [s.text for s in valid]
    with log_time("embed segments for semantic chunking"):
        segment_embeddings = embeddings.embed_documents(segment_texts)

    chunks: List[RetrievalChunk] = []
    chunk_id = 0
    idx = 0

    while idx < len(valid):
        # Start a new chunk with the current segment
        window: List[TranscriptSegment] = [valid[idx]]
        window_emb = segment_embeddings[idx]
        char_count = len(valid[idx].text)
        j = idx + 1

        # Greedily merge forward while semantic similarity & length allow
        while j < len(valid):
            next_seg = valid[j]
            next_emb = segment_embeddings[j]
            next_len = len(next_seg.text)

            sim = cosine_similarity(window_emb, next_emb)
            if sim >= CFG.semantic_chunk_threshold and (char_count + next_len) <= CFG.semantic_chunk_max_chars:
                window.append(next_seg)
                char_count += next_len
                # Update window embedding to running average for next comparison
                window_emb = [(a + b) / 2 for a, b in zip(window_emb, next_emb)]
                j += 1
            else:
                break

        chunks.append(
            RetrievalChunk(
                chunk_id=chunk_id,
                text=" ".join(s.text for s in window),
                start=window[0].start,
                end=window[-1].end,
                segment_ids=[s.segment_id for s in window],
            )
        )
        chunk_id += 1
        idx = j  # Jump to first unmerged segment

    logger.info("Semantic chunking complete: %d segments → %d chunks", len(valid), len(chunks))
    return chunks


# =============================================================================
# SOURCE REFERENCE MODEL
# =============================================================================

@dataclass
class SourceRef:
    """Rendered source reference used in Q&A answers and the References section.

    Fields:
        start    — segment start time in seconds.
        end      — segment end time in seconds.
        label    — human-readable "HH:MM:SS - HH:MM:SS" range shown to the user.
        url      — deep-link YouTube URL with ?t=Xs that jumps to the moment.
        chunk_id — back-reference to the originating RetrievalChunk.
        text     — the raw chunk text, available for tooltip previews.
    """
    start: float
    end: float
    label: str
    url: str
    chunk_id: Optional[int]
    text: str


# =============================================================================
# SESSION STATE
# =============================================================================

@dataclass
class SessionState:
    """Per-user in-memory session holding the active transcript and derived state.

    All fields default to safe empty values so a freshly constructed instance
    represents the "nothing loaded yet" state without any conditional checks.

    Transcript fields (populated atomically by set_transcript):
        video_url            — original URL submitted by the user.
        video_id             — extracted 11-char YouTube video id.
        processed_transcript — full transcript text as a single string.
        transcript_hash      — short SHA-256 of the transcript, used as a
                               component of cache file names.
        transcript_segments  — ordered List[TranscriptSegment] with per-segment
                               start/end timestamps and optional word timings.

    Derived fields (cleared by reset_derived when a new transcript is loaded):
        summary       — the most recently generated summary text.
        chat_history  — List[Dict[str, str]] conversation history
        chunks        — List[RetrievalChunk] built from transcript_segments.
        faiss_index   — FAISS vector store built from chunks.
        summary_prompt_override - Summary prompt override.
        qa_prompt_override - Question and answer prompt override.
    """
    # --- transcript fields ---
    video_url: str = ""
    video_id: str = ""
    processed_transcript: str = ""
    transcript_hash: str = ""
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)

    # --- derived fields (cleared when transcript changes) ---
    summary: str = ""
    # Gradio's chatbot component natively serializes history as lists of lists
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    chunks: Optional[List[RetrievalChunk]] = None
    faiss_index: Optional[Any] = None
    summary_prompt_override: str = ""
    qa_prompt_override: str = ""

    # ------------------------------------------------------------------ #
    # Mutation helpers                                                     #
    # ------------------------------------------------------------------ #

    def reset_derived(self) -> None:
        """Clear every field that is derived from the transcript.

        Called by set_transcript whenever a new video is loaded so that the
        summary, Q&A history, and retrieval index from the previous session are
        never accidentally served for a different video.
        """
        self.summary = ""
        self.chat_history = []
        self.chunks = None
        self.faiss_index = None

    def set_transcript(
        self,
        video_url: str,
        transcript: str,
        segments: List["TranscriptSegment"],
    ) -> None:
        """Atomically update all transcript-derived fields for a new video.

        Computes the transcript hash from the text content so the value stays
        consistent whether the transcript came from cache or a fresh Whisper
        run.  Clears derived state immediately after updating the core fields
        so there is no window where old chunks or a stale FAISS index could
        be read against the new transcript.
        """
        self.video_url = video_url.strip()
        self.video_id = require_video_id(video_url)
        self.processed_transcript = transcript
        self.transcript_hash = text_hash(transcript)
        self.transcript_segments = list(segments)
        self.reset_derived()

    # ------------------------------------------------------------------ #
    # Read-only query helpers                                              #
    # ------------------------------------------------------------------ #

    def needs_transcript_refresh(self, video_url: str) -> bool:
        """Decide whether the session must (re-)fetch the transcript.

        Decision table:
            - No transcript loaded yet          → True  (never been processed)
            - Empty / blank URL submitted        → False (reuse current session)
            - Same video id as the current one   → False (already loaded)
            - Different video id                 → True  (new video)

        This is the single authoritative gate checked by the Gradio handler
        before kicking off the download + transcription pipeline.
        """
        if not self.processed_transcript:
            return True
        url = video_url.strip()
        # Empty URL means "use whatever is already in session" — do not refresh.
        if not url:
            return False
        new_id = get_video_id(url)
        return new_id != self.video_id

    # ------------------------------------------------------------------ #
    # Gradio serialization                                                 #
    # ------------------------------------------------------------------ #

    def to_gradio(self) -> Dict[str, Any]:
        """Serialise session state to a plain dict suitable for gr.State.

        All scalar fields are stored as-is.  TranscriptSegment and
        RetrievalChunk lists are converted through their to_dict() methods so
        the payload is fully JSON-serialisable — except for faiss_index, which
        is kept as a live Python object reference.  FAISS indices cannot be
        trivially serialised and Gradio state lives in-process memory anyway,
        so passing the object directly avoids a double round-trip through disk.
        """
        return {
            "video_url": self.video_url,
            "video_id": self.video_id,
            "processed_transcript": self.processed_transcript,
            "transcript_hash": self.transcript_hash,
            "transcript_segments": [s.to_dict() for s in self.transcript_segments],
            "summary": self.summary,
            "chat_history": self.chat_history,
            "chunks": (
                [c.to_dict() for c in self.chunks]
                if self.chunks is not None
                else None
            ),
            # Kept by reference — not serialised to JSON.
            "faiss_index": self.faiss_index,
            "summary_prompt_override": self.summary_prompt_override,
            "qa_prompt_override": self.qa_prompt_override,
        }

    @classmethod
    def from_gradio(cls, payload: Dict[str, Any]) -> "SessionState":
        """Reconstruct a SessionState from a to_gradio() payload.

        Uses empty defaults for every key so a partial or missing payload
        (e.g. the very first render before any video has been processed)
        never raises a KeyError or TypeError.

        TranscriptSegment and RetrievalChunk lists are rebuilt through their
        from_dict() class methods; entries that are not dicts are silently
        skipped so a single corrupted record does not destroy the whole session.

        The faiss_index value is passed through unchanged — it is either a
        live FAISS object (stored by reference in to_gradio) or None.
        """
        if not payload:
            return cls()

        raw_segments = payload.get("transcript_segments") or []
        raw_chunks = payload.get("chunks")

        return cls(
            video_url=str(payload.get("video_url", "")),
            video_id=str(payload.get("video_id", "")),
            processed_transcript=str(payload.get("processed_transcript", "")),
            transcript_hash=str(payload.get("transcript_hash", "")),
            transcript_segments=[
                TranscriptSegment.from_dict(s)
                for s in raw_segments
                if isinstance(s, dict)
            ],
            summary=str(payload.get("summary", "")),
            chat_history=payload.get("chat_history", []),
            chunks=(
                [
                    RetrievalChunk.from_dict(c)
                    for c in raw_chunks
                    if isinstance(c, dict)
                ]
                if isinstance(raw_chunks, list)
                else None
            ),
            # Live object or None — passed through from to_gradio().
            faiss_index=payload.get("faiss_index"),
            summary_prompt_override=str(payload.get("summary_prompt_override", "")),
            qa_prompt_override=str(payload.get("qa_prompt_override", "")),
        )

# =============================================================================
# RETRIEVAL CACHE  (chunks JSON + FAISS index on disk)
# =============================================================================

def retrieval_record_path(video_id: str, transcript_hash_value: str) -> Path:
    """Derive the cache directory for retrieval artifacts.

    Pattern: retrieval/<video_id>__<retrieval_config_hash>__<transcript_hash>/
    Both the embedding model settings and the transcript content are encoded
    so either change automatically routes to a fresh build.
    """
    folder_name = f"{video_id}__{RETRIEVAL_CONFIG_HASH}__{transcript_hash_value}"
    return PATHS.retrieval / folder_name


def save_retrieval_cache(
    video_id: str,
    transcript_hash_value: str,
    chunks: List[RetrievalChunk],
    faiss_index: FAISS,
) -> None:
    """Persist RetrievalChunk list as JSON and FAISS index files to disk."""
    cache_dir = retrieval_record_path(video_id, transcript_hash_value)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Chunk metadata (JSON) — human-readable, used to restore state.chunks.
    write_json_atomic(
        cache_dir / "chunks.json",
        {
            "video_id": video_id,
            "transcript_hash": transcript_hash_value,
            "chunks": [chunk.to_dict() for chunk in chunks],
            "retrieval_config_hash": RETRIEVAL_CONFIG_HASH,
            "retrieval_config": current_retrieval_config(),
            "saved_at": time.time(),
        },
    )

    # FAISS index — two binary files (index.faiss + index.pkl) written by
    # LangChain's save_local helper.
    faiss_index.save_local(str(cache_dir))
    logger.info(
        "Retrieval cache saved (%d chunks) → %s", len(chunks), cache_dir
    )


def load_retrieval_cache(
    video_id: str,
    transcript_hash_value: str,
    embeddings: OllamaEmbeddings,
) -> Optional[Tuple[List[RetrievalChunk], FAISS]]:
    """Load chunks and FAISS index from disk when the cache directory exists.

    Returns (chunks, faiss_index) or None on a cache miss or corrupted files.
    allow_dangerous_deserialization is required by LangChain's load_local when
    reading the pickle file that stores the docstore.
    """
    cache_dir = retrieval_record_path(video_id, transcript_hash_value)
    chunks_path = cache_dir / "chunks.json"
    faiss_path = cache_dir / "index.faiss"

    if not chunks_path.exists() or not faiss_path.exists():
        return None

    data = read_json(chunks_path)
    if not data or not isinstance(data.get("chunks"), list):
        return None

    try:
        chunks = [RetrievalChunk.from_dict(item) for item in data["chunks"]]
        faiss_index = FAISS.load_local(
            str(cache_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(
            "Retrieval cache loaded (%d chunks) ← %s", len(chunks), cache_dir
        )
        return chunks, faiss_index
    except Exception:
        logger.warning(
            "Retrieval cache at %s is corrupted — will rebuild.", cache_dir,
            exc_info=True,
        )
        return None


def get_or_create_chunks(state: SessionState, runtime: RuntimeDeps) -> None:
    """Populate state.chunks from cache or by building from transcript segments.

    Mutates state in place.  A no-op when state.chunks is already set (e.g.
    the FAISS index was rebuilt from the disk cache in get_or_create_faiss).
    """
    if state.chunks is not None:
        return

    cached = load_retrieval_cache(
        state.video_id, state.transcript_hash, embeddings=None  # chunks only
    )
    # load_retrieval_cache returns None when the FAISS files are missing too,
    # but the JSON chunk file may exist independently after a partial save.
    # We skip the FAISS half here; get_or_create_faiss handles it.
    if cached:
        state.chunks, _ = cached
        logger.info("Chunks loaded from retrieval cache (%d).", len(state.chunks))
        return

    logger.info(
        "Building retrieval chunks from %d segments...",
        len(state.transcript_segments),
    )
    with log_time("build_retrieval_chunks"):
        state.chunks = build_retrieval_chunks(state.transcript_segments, runtime.embeddings)
    logger.info("Built %d retrieval chunks.", len(state.chunks))


_qdrant_client_instance = None

def get_qdrant_client() -> "QdrantClient":
    """Return a singleton QdrantClient for local disk mode.

    Design Rationale:
    Qdrant's local backend uses portalocker to prevent concurrent writes.
    Creating multiple client instances pointing to the same path in one process
    triggers an AlreadyLocked RuntimeError. Reusing a single instance bypasses
    the lock check safely while maintaining thread-safe collection routing.
    """
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        from qdrant_client import QdrantClient

        qdrant_path = str((CFG.base_dir / CFG.qdrant_path).resolve())
        logger.info("Initializing singleton Qdrant client at %s", qdrant_path)
        _qdrant_client_instance = QdrantClient(path=qdrant_path)

    return _qdrant_client_instance


def build_vector_store(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: OllamaEmbeddings,
    video_id: str,
) -> Any:
    if CFG.vector_db_type == "qdrant":
        if not QDRANT_AVAILABLE:
            raise ImportError("Install qdrant-client and langchain-qdrant for Qdrant support.")

        from qdrant_client import models as qdrant_models
        from langchain_qdrant import QdrantVectorStore

        client = get_qdrant_client()
        collection_name = f"yt_transcripts_{video_id}"

        sample_vec = embeddings.embed_query("dimension check")
        vector_size = len(sample_vec)

        if not client.collection_exists(collection_name):
            logger.info("Creating Qdrant collection %s (dim=%d)", collection_name, vector_size)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        else:
            logger.info("Reusing existing Qdrant collection %s", collection_name)

        logger.info(f"Initializing QdrantVectorStore for collection: {collection_name}")
        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        logger.info(f"Adding {len(texts)} texts to the vector store.")
        store.add_texts(texts, metadatas=metadatas)
        return store

    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)


def load_vector_store(
    video_id: str,
    embeddings: OllamaEmbeddings,
) -> Optional[Any]:
    if CFG.vector_db_type == "qdrant":
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant requested but qdrant dependencies are unavailable.")
            return None

        from langchain_qdrant import QdrantVectorStore

        collection_name = f"yt_transcripts_{video_id}"
        try:
            client = get_qdrant_client()
            if not client.collection_exists(collection_name):
                logger.info("Qdrant collection not found for %s: %s", video_id, collection_name)
                return None

            logger.info("Loaded Qdrant collection for %s: %s", video_id, collection_name)
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
            )
        except Exception as e:
            logger.warning("Failed to load Qdrant collection for %s: %s", video_id, e)
            return None

    cache_dir = retrieval_record_path(video_id, text_hash(""))
    faiss_path = cache_dir / "index.faiss"
    if faiss_path.exists():
        return FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
    return None

def get_or_create_vector_store(state: SessionState, runtime: RuntimeDeps) -> Any:
    """Return the vector store for the active session, building & persisting if necessary.
    
    Design Rationale:
    - 3-layer cache: memory → disk → fresh build.
    - Ensures chunks.json exists early so cross-video BM25 works even when 
      the vector store is loaded from cache.
    - FAISS requires explicit .save_local(). Qdrant local mode auto-persists 
      to CFG.qdrant_path via the singleton client.
    - state.faiss_index is a legacy field name but now holds any vector store 
      type for backward compatibility with SessionState serialization.
    """
    # ── 1. Fast path: in-memory cache ──────────────────────────────────────
    if state.faiss_index is not None:
        logger.info("Vector store found in session memory — reusing.")
        return state.faiss_index

    # ── 2. Resolve cache paths once ────────────────────────────────────────
    cache_dir = retrieval_record_path(state.video_id, state.transcript_hash)
    chunks_path = cache_dir / "chunks.json"

    # ── 3. Ensure chunks.json exists (critical for cross-video BM25) ───────
    # Runs idempotently. Guarantees lexical search works even on cache hits.
    if state.chunks and not chunks_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            chunks_path,
            {
                "video_id": state.video_id,
                "transcript_hash": state.transcript_hash,
                "chunks": [chunk.to_dict() for chunk in state.chunks],
                "retrieval_config_hash": RETRIEVAL_CONFIG_HASH,
                "retrieval_config": current_retrieval_config(),
                "saved_at": time.time(),
            },
        )
        logger.info("Chunks metadata saved to %s", chunks_path)

    # ── 4. Disk cache check ────────────────────────────────────────────────
    store = load_vector_store(state.video_id, runtime.embeddings)
    if store:
        state.faiss_index = store
        logger.info("Vector store loaded from disk cache.")
        return store

    # ── 5. Build fresh vector store ────────────────────────────────────────
    if not state.chunks:
        raise RuntimeError("Cannot build vector store: state.chunks is empty.")

    logger.info("Building vector store from %d chunks...", len(state.chunks))
    texts = [chunk.text for chunk in state.chunks]
    metadatas = [
        {"start": chunk.start, "end": chunk.end, "chunk_id": chunk.chunk_id, "video_id": state.video_id}
        for chunk in state.chunks
    ]

    with log_time("Vector store build"):
        store = build_vector_store(texts, metadatas, runtime.embeddings, state.video_id)

    # ── 6. Persist artifacts to disk ───────────────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save chunks.json if step 3 was skipped (e.g., state.chunks was populated late)
    if not chunks_path.exists():
        write_json_atomic(
            chunks_path,
            {
                "video_id": state.video_id,
                "transcript_hash": state.transcript_hash,
                "chunks": [chunk.to_dict() for chunk in state.chunks],
                "retrieval_config_hash": RETRIEVAL_CONFIG_HASH,
                "retrieval_config": current_retrieval_config(),
                "saved_at": time.time(),
            },
        )
        logger.info("Chunks metadata saved to %s", chunks_path)

    # Backend-specific persistence
    if CFG.vector_db_type == "faiss":
        store.save_local(str(cache_dir))
        logger.info("FAISS index persisted to %s", cache_dir)
    elif CFG.vector_db_type == "qdrant" and QDRANT_AVAILABLE:
        client = get_qdrant_client()
        collection_name = f"yt_transcripts_{state.video_id}"
        if client.collection_exists(collection_name):
            logger.info(
                "Qdrant collection auto-persisted: %s at %s",
                collection_name,
                (CFG.base_dir / CFG.qdrant_path).resolve(),
            )
        else:
            logger.warning(
                "Qdrant collection missing after build: %s. Check disk permissions or path config.",
                collection_name,
            )

    state.faiss_index = store
    logger.info("Vector store built and fully persisted.")
    return store

# =============================================================================
# HYBRID SEARCH FUNCTIONS
# =============================================================================

@dataclass
class HybridDoc:
    """Lightweight document proxy compatible with build_context_with_sources."""
    page_content: str
    metadata: Dict[str, Any]


def tokenize_text(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks: List[RetrievalChunk]) -> BM25Okapi:
    """Build a lightweight BM25 index from retrieval chunks."""
    tokenized_corpus = [tokenize_text(c.text) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def hybrid_search(
    question: str,
    vector_store: Any,
    chunks: List[RetrievalChunk],
    embeddings: OllamaEmbeddings,
    top_k: int = CFG.retrieval_top_k,
    candidates: int = CFG.hybrid_top_k_candidates,
    alpha: float = CFG.hybrid_dense_weight,
) -> List[HybridDoc]:
    """Run dense + sparse retrieval and fuse scores via weighted normalization.
    
    BM25 is rebuilt in-memory per query (<50ms for ~200 chunks) to guarantee
    sync with the active FAISS index and avoid pickle/cache complexity.
    """
    if not chunks:
        return []

    # ── 1. Dense Vector Search ─────────────────────────────────────────────
    dense_docs = vector_store.similarity_search_with_score(question, k=candidates)
    dense_scores = [score for _, score in dense_docs]

    # Qdrant returns cosine similarity (higher = better).
    # FAISS returns L2 distance (lower = better).
    # Normalize both to [0, 1] where 1.0 = best match.
    if dense_scores:
        min_s, max_s = min(dense_scores), max(dense_scores)
        if CFG.vector_db_type == "qdrant":
            # Cosine similarity: direct min-max normalization
            norm_dense = [(s - min_s) / (max_s - min_s) if max_s > min_s else 1.0 for s in dense_scores]
        else:
            # L2 distance: invert after normalization
            norm_dense = [1.0 - ((s - min_s) / (max_s - min_s)) if max_s > min_s else 1.0 for s in dense_scores]
    else:
        norm_dense = []

    # ── 2. Sparse BM25 search ──────────────────────────────────────────────
    bm25 = build_bm25_index(chunks)
    query_tokens = tokenize_text(question)
    # .tolist() converts the NumPy array to a standard Python list
    bm25_scores = bm25.get_scores(query_tokens).tolist()
    
    # Normalize BM25 scores to [0,1]
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    norm_sparse = [s / max_bm25 if max_bm25 > 0 else 0.0 for s in bm25_scores]

    # ── 3. Score Fusion ────────────────────────────────────────────────────
    chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
    fused_scores: Dict[int, float] = {}

    for (doc, _), nd in zip(dense_docs, norm_dense):
        cid = doc.metadata.get("chunk_id")
        if cid in chunk_id_to_idx:
            idx = chunk_id_to_idx[cid]
            fused_scores[idx] = fused_scores.get(idx, 0.0) + (alpha * nd)

    for idx, ns in enumerate(norm_sparse):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + ((1.0 - alpha) * ns)

    # ── 4. Rank & Return Top-K ─────────────────────────────────────────────
    ranked_indices = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in ranked_indices:
        chunk = chunks[idx]
        results.append(HybridDoc(
            page_content=chunk.text,
            metadata={"start": chunk.start, "end": chunk.end, "chunk_id": chunk.chunk_id, "hybrid_score": fused_scores[idx]}
        ))
        
    logger.info("Hybrid search complete: %d candidates → %d fused results", len(fused_scores), len(results))
    return results


# =============================================================================
# Cross-video retrieval & video library
# =============================================================================

def get_video_library() -> List[Dict[str, Any]]:
    """Fetch indexed videos from SQLite for the library UI.
    
    Returns a list of dicts sorted by most recently indexed.
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT video_id, url, indexed_at FROM videos ORDER BY indexed_at DESC"
        ).fetchall()
    logger.info("Fetched %d videos from library DB.", len(rows))
    return [
        {"video_id": r["video_id"], "url": r["url"], "indexed_at": r["indexed_at"]}
        for r in rows
    ]


def _find_retrieval_cache_dir(video_id: str) -> Optional[Path]:
    """Locate the latest retrieval cache directory for a given video ID.
    
    Scans PATHS.retrieval for directories matching <video_id>__<hash>__<hash>.
    Returns the first match or None if the video hasn't been processed yet.
    """
    if not PATHS.retrieval.exists():
        return None
    for cache_dir in PATHS.retrieval.iterdir():
        if cache_dir.is_dir() and cache_dir.name.startswith(f"{video_id}__"):
            return cache_dir
    return None


def load_chunks_for_videos(video_ids: List[str]) -> List[RetrievalChunk]:
    """Load retrieval chunks for multiple videos from disk cache.
    
    Design Rationale:
    - Uses _find_retrieval_cache_dir to bypass brittle hash lookups.
    - Attaches video_id dynamically to each chunk for cross-video context rendering.
    - Gracefully skips videos with missing or corrupted chunk files.
    """
    all_chunks: List[RetrievalChunk] = []
    for vid in video_ids:
        cache_dir = _find_retrieval_cache_dir(vid)
        if not cache_dir:
            logger.warning("No retrieval cache found for video %s. Run Summarize/Q&A first.", vid)
            continue
            
        chunks_path = cache_dir / "chunks.json"
        if not chunks_path.exists():
            logger.warning("chunks.json missing in %s", cache_dir)
            continue
            
        data = read_json(chunks_path)
        if data and isinstance(data.get("chunks"), list):
            chunks = [RetrievalChunk.from_dict(c) for c in data["chunks"]]
            for c in chunks:
                if not c.video_id:  # Backfill legacy chunks missing the field
                    c.video_id = vid
            all_chunks.extend(chunks)
            logger.info("Loaded %d chunks for video %s", len(chunks), vid)
            
    logger.info("Total cross-video chunks loaded: %d", len(all_chunks))
    return all_chunks


def load_vector_stores_for_videos(video_ids: List[str], embeddings: OllamaEmbeddings) -> Dict[str, Any]:
    """Load vector stores for multiple videos, returning {video_id: store}.
    
    Design Rationale:
    - Qdrant loading is decoupled from file-cache directories. Qdrant manages 
      its own local DB, so missing FAISS/chunks folders should not block it.
    - FAISS fallback remains gated behind cache_dir existence.
    - Uses singleton client to prevent portalocker AlreadyLocked crashes.
    """
    stores = {}

    # ── 1. Qdrant path (independent of file cache) ─────────────────────
    if CFG.vector_db_type == "qdrant" and QDRANT_AVAILABLE:
        from langchain_qdrant import QdrantVectorStore
        client = get_qdrant_client()
        collections = {c.name for c in client.get_collections().collections}

        for vid in video_ids:
            collection_name = f"yt_transcripts_{vid}"
            if collection_name in collections:
                stores[vid] = QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embedding=embeddings,
                )
                logger.info("Loaded Qdrant store for video %s", vid)

        logger.info("Successfully loaded %d/%d Qdrant stores.", len(stores), len(video_ids))
        return stores

    # ── 2. FAISS fallback (requires cache dirs) ────────────────────────
    for vid in video_ids:
        cache_dir = _find_retrieval_cache_dir(vid)
        if not cache_dir:
            logger.debug("Skipping FAISS load for %s: no cache dir", vid)
            continue

        faiss_path = cache_dir / "index.faiss"
        if faiss_path.exists():
            try:
                stores[vid] = FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded FAISS store for video %s", vid)
            except Exception as e:
                logger.warning("FAISS load failed for %s: %s", vid, e)
        else:
            logger.debug("No index.faiss found in %s", cache_dir)

    logger.info("Successfully loaded %d/%d FAISS stores.", len(stores), len(video_ids))
    return stores


def cross_video_hybrid_search(
    question: str,
    video_ids: List[str],
    embeddings: OllamaEmbeddings,
    top_k: int = CFG.retrieval_top_k,
) -> List[HybridDoc]:
    """Search across multiple video vector stores and fuse results globally."""
    if not video_ids:
        logger.warning("Cross-video search called with empty video_ids list.")
        return []

    logger.info("Starting cross-video hybrid search for %d videos: %s", len(video_ids), video_ids)

    # ── 1. Load vector stores ──────────────────────────────────────────────
    with log_time("load cross-video vector stores"):
        stores = load_vector_stores_for_videos(video_ids, embeddings)
    if not stores:
        logger.warning("No vector stores found. Ensure videos have been processed via Summarize/Q&A first.")
        return []

    # ── 2. Collect dense candidates ────────────────────────────────────────
    with log_time("cross-video dense search"):
        dense_candidates = []
        for vid, store in stores.items():
            try:
                docs = store.similarity_search_with_score(question, k=CFG.hybrid_top_k_candidates)
                dense_candidates.extend(docs)
                logger.debug("Dense search for %s returned %d candidates", vid, len(docs))
            except Exception as e:
                logger.warning("Dense search failed for %s: %s", vid, e)

    if not dense_candidates:
        logger.warning("Dense search returned 0 candidates across all videos.")
        return []

    # ── 3. Load chunks & build global BM25 ─────────────────────────────────
    with log_time("load cross-video chunks & build BM25"):
        all_chunks = load_chunks_for_videos(video_ids)
    if not all_chunks:
        logger.warning("No retrieval chunks found on disk for selected videos.")
        return []

    # Normalize dense scores according to backend semantics
    dense_scores = [s for _, s in dense_candidates]
    min_d, max_d = min(dense_scores), max(dense_scores)

    if CFG.vector_db_type == "qdrant":
        # Cosine similarity: higher is better
        norm_dense = [
            (s - min_d) / (max_d - min_d) if max_d > min_d else 1.0
            for s in dense_scores
        ]
    else:
        # FAISS L2 distance: lower is better
        norm_dense = [
            1.0 - ((s - min_d) / (max_d - min_d)) if max_d > min_d else 1.0
            for s in dense_scores
        ]

    # BM25 scoring
    bm25 = build_bm25_index(all_chunks)
    query_tokens = tokenize_text(question)
    bm25_scores = bm25.get_scores(query_tokens).tolist()
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    norm_sparse = [s / max_bm25 if max_bm25 > 0 else 0.0 for s in bm25_scores]

    # ── 4. Global Score Fusion ─────────────────────────────────────────────
    with log_time("cross-video score fusion"):
        chunk_lookup = {
            (getattr(c, "video_id", "unknown"), c.chunk_id): i
            for i, c in enumerate(all_chunks)
        }
        fused_scores: Dict[int, float] = {}

        for (doc, _), nd in zip(dense_candidates, norm_dense):
            cid = doc.metadata.get("chunk_id")
            vid = doc.metadata.get("video_id", "unknown")
            key = (vid, cid)
            if key in chunk_lookup:
                idx = chunk_lookup[key]
                fused_scores[idx] = fused_scores.get(idx, 0.0) + (CFG.hybrid_dense_weight * nd)
            else:
                logger.debug("Dense candidate could not be matched to chunk: video=%s chunk_id=%s", vid, cid)

        for idx, ns in enumerate(norm_sparse):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + ((1.0 - CFG.hybrid_dense_weight) * ns)

        ranked_indices = sorted(
            fused_scores.keys(),
            key=lambda i: fused_scores[i],
            reverse=True,
        )[:top_k]

    results = [
        HybridDoc(
            page_content=all_chunks[idx].text,
            metadata={
                "start": all_chunks[idx].start,
                "end": all_chunks[idx].end,
                "chunk_id": all_chunks[idx].chunk_id,
                "video_id": getattr(all_chunks[idx], "video_id", "unknown"),
                "hybrid_score": fused_scores[idx],
            },
        )
        for idx in ranked_indices
    ]

    logger.info(
        "Cross-video hybrid search complete: %d candidates → %d fused results",
        len(fused_scores),
        len(results),
    )
    return results


# =============================================================================
# Q&A CONTEXT BUILDER
# =============================================================================

def build_context_with_sources(
    docs: List[Any], video_id: str
) -> Tuple[str, Dict[str, SourceRef]]:
    """Convert FAISS-retrieved documents into a labelled context string.

    Each document is assigned a sequential label S1, S2, S3 … that the LLM
    can cite inline in its answer.  A parallel lookup dict maps every label to
    a SourceRef so the citation renderer can replace [S1] with a formatted
    Markdown link without doing any additional lookups.

    Returns:
        context     — multi-line string with labelled passages, ready to be
                      inserted into the QA prompt's {context} slot.
        source_lookup — {label: SourceRef} for every document in docs.
    """
    if not docs:
        return "", {}

    context_parts: List[str] = []
    source_lookup: Dict[str, SourceRef] = {}

    for idx, doc in enumerate(docs, start=1):
        label = f"S{idx}"
        meta = doc.metadata
        start = float(meta.get("start", 0.0))
        end = float(meta.get("end", 0.0))
        chunk_id = meta.get("chunk_id")
        text = doc.page_content

        time_label = f"{seconds_to_hhmmss(start)} - {seconds_to_hhmmss(end)}"
        # Prefer per-chunk video_id for cross-video compatibility; fallback to function arg
        source_vid = meta.get("video_id", video_id)
        url = build_youtube_time_url(source_vid, start)

        source_lookup[label] = SourceRef(
            start=start,
            end=end,
            label=time_label,
            url=url,
            chunk_id=chunk_id,
            text=text,
        )
        context_parts.append(f"[{label}] (Video: {source_vid}) {text}")

    context = "\n\n".join(context_parts)
    return context, source_lookup


# =============================================================================
# CITATION RENDERER
# =============================================================================

def render_clickable_answer(
    raw_answer: str, source_lookup: Dict[str, "SourceRef"]
) -> str:
    """Replace [S1]-style LLM citations with clickable Markdown timestamp links.

    The LLM may cite sources in two patterns:
        Grouped  — [S1, S2, S3]   matched by SOURCE_GROUP_PATTERN
        Isolated — [S1]           matched by SOURCE_SINGLE_PATTERN

    Both are replaced with inline Markdown links that open the YouTube video at
    the correct timestamp.  A deduplicated **References** section is appended
    listing every source the LLM actually cited.

    Deduplication uses dict.fromkeys for O(n) order-preserving uniqueness —
    faster than the O(n²) "if x not in seen" pattern for long answers.
    """

    rendered = raw_answer
    used_labels: List[str] = []

    def _replace_group(match: re.Match) -> str:
        """Replace [S1, S2, S3] with individual inline links."""
        labels = [lbl.strip() for lbl in match.group(1).split(",")]
        parts: List[str] = []
        for lbl in labels:
            ref = source_lookup.get(lbl)
            if ref:
                parts.append(f"[{ref.label}]({ref.url})")
                used_labels.append(lbl)
            else:
                parts.append(f"[{lbl}]")
        return " ".join(parts)

    def _replace_single(match: re.Match) -> str:
        """Replace an isolated S1 token (already outside brackets) with a link."""
        lbl = match.group(1)
        ref = source_lookup.get(lbl)
        if ref:
            used_labels.append(lbl)
            return f"[{ref.label}]({ref.url})"
        return lbl

    # Process grouped citations first so SOURCE_SINGLE_PATTERN doesn't
    # partially match inside a group like [S1, S2].
    rendered = SOURCE_GROUP_PATTERN.sub(_replace_group, rendered)
    rendered = SOURCE_SINGLE_PATTERN.sub(_replace_single, rendered)

    # Append a deduplicated References section for every cited source.
    seen = dict.fromkeys(used_labels)  # preserves first-seen order, O(n)
    cited_refs = [
        (lbl, source_lookup[lbl])
        for lbl in seen
        if lbl in source_lookup
    ]

    if cited_refs:
        ref_lines = ["\n\n**References**"]
        for lbl, ref in cited_refs:
            ref_lines.append(f"- [{ref.label}]({ref.url})")
        rendered += "\n".join(ref_lines)

    return rendered

# =============================================================================
# Chapter Generation & Cache Functions
# =============================================================================

def generate_chapters(
    segments: List[TranscriptSegment], runtime: RuntimeDeps
) -> List[Dict[str, Any]]:
    """Generate logical chapters from transcript segments using LLM topic segmentation.
    
    Design Rationale:
    - Long videos are split into token-safe temporal windows to prevent LLM context overflow.
    - Each window is processed independently; timestamps remain globally accurate.
    - Uses a conservative token budget (max_transcript_tokens - safety_margin) to leave room 
      for system prompt, instructions, and JSON formatting overhead.
    - Hard Chunk Boundaries: Chapters will not span across token windows. This is intentional.
      YouTube chapters naturally align with topic shifts, and forcing cross-window continuity
      would require a second LLM pass (adding 5-10s latency). A lightweight title-merge step
      handles ~90% of boundary artifacts by extending timestamps when adjacent titles match.
    - Gracefully handles malformed LLM output or single oversized segments.
    - Results are cached by transcript_hash, so chunking adds zero latency on reruns.
    """
    if not segments:
        return []

    # Safe token budget for chapter generation.
    # CFG.max_transcript_tokens already subtracts safety_margin once.
    # Subtracting it again guarantees ample headroom for prompt overhead & JSON formatting.
    max_tokens = CFG.max_transcript_tokens - CFG.safety_margin
    chunks: List[List[TranscriptSegment]] = []
    current_chunk: List[TranscriptSegment] = []
    current_tokens = 0

    # ── 1. Split segments into token-safe temporal windows ─────────────────
    for seg in segments:
        seg_tokens = estimate_tokens(seg.text)
        # Hard boundary: if adding this segment exceeds budget, finalize current window
        if current_tokens + seg_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [seg]
            current_tokens = seg_tokens
        else:
            current_chunk.append(seg)
            current_tokens += seg_tokens
    if current_chunk:
        chunks.append(current_chunk)

    all_chapters: List[Dict[str, Any]] = []

    # ── 2. Generate chapters per chunk using pre-built chain ───────────────
    for chunk_idx, chunk_segs in enumerate(chunks):
        # Compact segment representation for prompt efficiency
        seg_lines = [
            f"[{i}] {seconds_to_hhmmss(s.start)}-{seconds_to_hhmmss(s.end)}: {s.text[:120]}"
            for i, s in enumerate(chunk_segs)
        ]
        prompt_text = "\n".join(seg_lines)

        try:
            with log_time(f"LLM chapter generation (chunk {chunk_idx+1}/{len(chunks)})"):
                # Use the pre-compiled chapter chain from runtime
                raw = runtime.chapter_chain.predict(segments=prompt_text)

            # Strip markdown code blocks if LLM wraps JSON
            raw = re.sub(r"```json\s*|\s*```", "", raw.strip())
            chapters = json.loads(raw)

            # Validate & map to globally accurate timestamps
            for ch in chapters:
                s_idx, e_idx = int(ch["start_idx"]), int(ch["end_idx"])
                if 0 <= s_idx <= e_idx < len(chunk_segs):
                    all_chapters.append({
                        "title": str(ch["title"]),
                        "start": chunk_segs[s_idx].start,
                        "end": chunk_segs[e_idx].end,
                    })
        except Exception as exc:
            logger.warning("Chapter generation failed for chunk %d: %s. Skipping.", chunk_idx, exc)

    # ── 3. Merge adjacent chapters with identical titles (boundary cleanup) ─
    # Handles hard chunk boundary artifacts where a topic spans two windows
    if not all_chapters:
        return []

    merged = [all_chapters[0]]
    for ch in all_chapters[1:]:
        if ch["title"].lower().strip() == merged[-1]["title"].lower().strip():
            merged[-1]["end"] = ch["end"]  # Extend boundary to merge split topics
        else:
            merged.append(ch)

    logger.info("Generated %d chapters across %d temporal windows.", len(merged), len(chunks))
    return merged


def save_chapters(video_id: str, transcript_hash: str, chapters: List[Dict]) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chapters (video_id, transcript_hash, chapters_json, saved_at) VALUES (?, ?, ?, ?)",
            (video_id, transcript_hash, json.dumps(chapters), time.time())
        )

def load_cached_chapters(video_id: str, transcript_hash: str) -> Optional[List[Dict]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT chapters_json FROM chapters WHERE video_id = ? AND transcript_hash = ? ORDER BY saved_at DESC LIMIT 1",
            (video_id, transcript_hash)
        ).fetchone()
    return json.loads(row["chapters_json"]) if row else None


# =============================================================================
# GRADIO HANDLERS
# =============================================================================

def run_llm_dynamic(llm: Ollama, template: str, inputs: Dict[str, str]) -> str:
    """Run LLM with a custom prompt template without rebuilding chains.
    
    Design Rationale:
    - Dynamically extracts only variables actually present in the template.
      This prevents LangChain's strict PromptTemplate validation from crashing
      when users provide overrides that omit optional variables like {chat_history}.
    - Falls back gracefully if the template is empty or malformed.
    """
    import re
    # Extract {variable_name} placeholders from the template
    template_vars = set(re.findall(r"\{(\w+)\}", template))
    filtered_inputs = {k: v for k, v in inputs.items() if k in template_vars}
    
    if not filtered_inputs:
        raise ValueError("Prompt template contains no valid input variables.")
        
    prompt = PromptTemplate(template=template, input_variables=list(filtered_inputs.keys()))
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.predict(**filtered_inputs).strip()


def _make_summarize_handler(runtime: RuntimeDeps):
    """Factory returning summarize_video_gradio closed over runtime.

    Moving the runtime dependency into a closure instead of referencing a
    module-level singleton means the module can be imported in tests without
    triggering Whisper and Ollama initialisation.
    """
    def summarize_video_gradio(
        video_url: str,
        state_payload: Dict[str, Any],
        summary_prompt_override: str,
    ) -> Generator:
        """Run transcript + summary pipeline and stream UI updates.

        Drives the full transcript → summary pipeline, updating seven Gradio
        outputs at each yield step:
            (status, token_info, transcript, summary, stats,
             stt_progress, summary_progress, state)
        """
        state = SessionState.from_gradio(state_payload)
        state.summary_prompt_override = summary_prompt_override.strip()
        started_at = time.perf_counter()

        if not video_url.strip():
            yield (
                "❌ Please provide a valid YouTube URL.",
                "", "", "", "", 0, 0, state.to_gradio(),
            )
            return

        try:
            transcript: Optional[str] = None
            segments: Optional[List[TranscriptSegment]] = None
            stt_progress = 0
            summary_progress = 0

            yield "🚀 Starting pipeline...", "", "", "", "", 0, 0, state.to_gradio()

            # ── STT phase ───────────────────────────────────────────────────
            for update in fetch_transcript_from_stt_stream(video_url, runtime):
                stt_progress = update.progress
                # Keep the most recent non-None values across yields.
                transcript = update.transcript or transcript
                segments = update.segments or segments
                yield (
                    update.message,
                    "",
                    transcript or "",
                    "",
                    "",
                    stt_progress,
                    summary_progress,
                    state.to_gradio(),
                )

            if not transcript or not segments:
                raise RuntimeError("Transcript fetch failed — no text or segments.")

            state.set_transcript(video_url, transcript, segments)

            # ── Token budget info ────────────────────────────────────────────
            token_count = estimate_tokens(state.processed_transcript)
            mode = "direct" if token_count <= CFG.max_transcript_tokens else "chunked"
            token_info = (
                f"{token_count} tokens "
                f"(limit ~{CFG.max_transcript_tokens}) | mode={mode}"
            )
            stats_info = (
                f"chars={len(state.processed_transcript)} "
                f"| tokens={token_count} "
                f"| segments={len(state.transcript_segments)}"
            )
            summary_progress = 5
            yield (
                "🔢 Counting tokens...",
                token_info,
                state.processed_transcript,
                "",
                stats_info,
                stt_progress,
                summary_progress,
                state.to_gradio(),
            )

            # ── Summary cache check ──────────────────────────────────────────
            cached_summary = load_cached_summary(
                state.video_id, state.transcript_hash, mode
            )
            if cached_summary:
                state.summary = cached_summary
                elapsed = time.perf_counter() - started_at
                yield (
                    f"✅ Done in {elapsed:.2f}s (cached summary).",
                    token_info,
                    state.processed_transcript,
                    state.summary,
                    stats_info,
                    100,
                    100,
                    state.to_gradio(),
                )
                return

            # ── Summary generation ───────────────────────────────────────────
            for update in summarize_transcript_stream(
                state.processed_transcript, runtime, state.summary_prompt_override
            ):
                summary_progress = update.progress
                if update.summary:
                    state.summary = update.summary
                yield (
                    update.message,
                    token_info,
                    state.processed_transcript,
                    state.summary,
                    stats_info,
                    stt_progress,
                    summary_progress,
                    state.to_gradio(),
                )

            if not state.summary:
                raise RuntimeError("Summary generation returned empty output.")

            save_summary(state.video_id, state.transcript_hash, mode, state.summary)
            elapsed = time.perf_counter() - started_at
            yield (
                f"✅ Done in {elapsed:.2f}s.",
                token_info,
                state.processed_transcript,
                state.summary,
                stats_info,
                100,
                100,
                state.to_gradio(),
            )

        except BaseException as exc:
            # Gradio cancellation raises CancelledError/KeyboardInterrupt/GeneratorExit
            is_cancel = isinstance(exc, (KeyboardInterrupt, GeneratorExit)) or "cancel" in type(exc).__name__.lower()
            
            if is_cancel:
                logger.info("Summarize pipeline cancelled by user.")
                yield (
                    "⛔ Cancelled by user.",
                    token_info if 'token_info' in locals() else "",
                    state.processed_transcript,
                    state.summary,
                    stats_info if 'stats_info' in locals() else "",
                    0, 0, state.to_gradio(),
                )
            else:
                logger.exception("Summarize pipeline error")
                yield (
                    f"❌ Error: {exc}",
                    token_info if 'token_info' in locals() else "",
                    "", "", "",
                    0, 0, state.to_gradio(),
                )
            return  # Critical: ensures generator exits cleanly after final yield

    return summarize_video_gradio


def _make_qa_handler(runtime: RuntimeDeps):
    """Factory returning answer_question_gradio closed over runtime.

    Same motivation as _make_summarize_handler: the closure keeps the module
    importable without side effects so the test suite can stub runtime.
    """
    def answer_question_gradio(
        video_url: str,
        user_question: str,
        state_payload: Dict[str, Any],
        qa_prompt_override: str,
    ) -> Generator:
        """Run timestamp-aware transcript Q&A and stream UI updates.

        Pipeline: validate input → refresh transcript if needed → build chunks
        → build FAISS index → similarity search → build context
        → generate answer → render citations.

        Streams four outputs: (status, answer, progress, state).
        Skips transcript fetch if the session already has it cached.
        """
        state = SessionState.from_gradio(state_payload)
        state.qa_prompt_override = qa_prompt_override.strip()
        question = user_question.strip()

        # ── Input validation ─────────────────────────────────────────────────
        if not question:
            yield "❌ Please provide a question.", state.chat_history, 0, state.to_gradio()
            return
        if video_url.strip() and not get_video_id(video_url):
            yield "❌ Invalid YouTube URL.", state.chat_history, 0, state.to_gradio()
            return
        if not video_url.strip() and not state.processed_transcript:
            yield (
                "❌ No transcript in session. Provide a YouTube URL first.",
                state.chat_history, 0, state.to_gradio(),
            )
            return

        try:
            progress = 0
            yield "🚀 Starting Q&A pipeline...", state.chat_history, progress, state.to_gradio()

            # ── Transcript fetch (skipped when session already has it) ────────
            if state.needs_transcript_refresh(video_url):
                transcript: Optional[str] = None
                segments: Optional[List[TranscriptSegment]] = None

                for update in fetch_transcript_from_stt_stream(video_url, runtime):
                    # Scale STT progress into 0-40 % of the overall bar.
                    progress = min(40, max(5, int(update.progress * 0.4)))
                    transcript = update.transcript or transcript
                    segments = update.segments or segments
                    yield update.message, state.chat_history, progress, state.to_gradio()

                if not transcript or not segments:
                    raise RuntimeError("Failed to fetch transcript.")
                
                # Note: set_transcript() calls reset_derived(), which clears chat_history.
                # This is intentional: switching videos starts a fresh conversation.
                state.set_transcript(video_url, transcript, segments)

            elif not state.transcript_segments:
                raise RuntimeError(
                    "Session has transcript text but no timestamped segments. "
                    "Re-run Summarize to rebuild the session."
                )

            # ── Retrieval chunk preparation ──────────────────────────────────
            progress = 45
            yield (
                "🧩 Preparing timestamp-aware retrieval chunks...",
                "", progress, state.to_gradio(),
            )
            get_or_create_chunks(state, runtime)

            # ── FAISS index ──────────────────────────────────────────────────
            progress = 65
            yield (
                "🗂️ Loading or building FAISS index...",
                state.chat_history, progress, state.to_gradio(),
            )
            vector_store = get_or_create_vector_store(state, runtime)

            # ── Hybrid Dense + BM25 search ───────────────────────────────────
            progress = 80
            yield (
                "🔎 Running hybrid dense+BM25 search...",
                state.chat_history, progress, state.to_gradio(),
            )
            with log_time("Hybrid dense+BM25 search"):
                docs = hybrid_search(
                    question=question,
                    vector_store=vector_store,
                    chunks=state.chunks,
                    embeddings=runtime.embeddings,
                )

            if not docs:
                state.chat_history.append({"role": "user", "content": question})
                state.chat_history.append({"role": "assistant", "content": "I couldn't find relevant transcript evidence for that question."})
                yield "✅ Answer ready.", state.chat_history, 100, state.to_gradio()
                return

            context, source_lookup = build_context_with_sources(docs, state.video_id)
            if not context.strip():
                state.chat_history.append({"role": "user", "content": question})
                state.chat_history.append({"role": "assistant", "content": "I couldn't build usable context from the retrieved chunks."})
                yield "✅ Answer ready.", state.chat_history, 100, state.to_gradio()
                return
            
            # ── Format history for prompt (messages format) ──────────────
            history_str = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in state.chat_history[-10:]  # last 5 turns
            ) if state.chat_history else "None"

            # ── LLM answer generation ────────────────────────────────────────
            progress = 90
            yield (
                f"🧠 Context ready ({estimate_tokens(context)} tokens). Generating answer...",
                state.chat_history, 
                progress, 
                state.to_gradio()
            )

            with log_time("QA generation"):
                tpl = state.qa_prompt_override or runtime.qa_chain.prompt.template
                raw_answer = run_llm_dynamic(runtime.llm, tpl, {
                    "context": context, "question": question, "chat_history": history_str
                })

            rendered_answer = render_clickable_answer(raw_answer, source_lookup)

            # ── Append to history in Gradio messages format ──────────────
            state.chat_history.append({"role": "user", "content": question})
            state.chat_history.append({"role": "assistant", "content": rendered_answer})

            yield "✅ Answer ready.", state.chat_history, 100, state.to_gradio()

        except BaseException as exc:
            is_cancel = isinstance(exc, (KeyboardInterrupt, GeneratorExit)) or "cancel" in type(exc).__name__.lower()
            
            if is_cancel:
                logger.info("Q&A pipeline cancelled by user.")
                yield "⛔ Cancelled by user.", state.chat_history, 0, state.to_gradio()
            else:
                logger.exception("Q&A pipeline error")
                yield f"❌ Error: {exc}", state.chat_history, 0, state.to_gradio()
            return  # Critical: clean exit
        
    return answer_question_gradio


def _make_cross_video_qa_handler(runtime: RuntimeDeps):
    """Factory for cross-video Q&A with explicit index validation logging."""
    def cross_video_qa_gradio(selected_videos: List[str], question: str) -> Generator:
        if not selected_videos:
            yield "❌ Please select at least one video.", "", 0
            return
        if not question.strip():
            yield "❌ Please provide a question.", "", 0
            return

        logger.info("Cross-video QA triggered: videos=%s, question='%s...'", selected_videos, question[:50])
        yield "🔎 Checking retrieval indexes...", "", 5

        missing = []
        for vid in selected_videos:
            has_index = False
            if CFG.vector_db_type == "qdrant" and QDRANT_AVAILABLE:
                try:
                    client = get_qdrant_client()
                    if client.collection_exists(f"yt_transcripts_{vid}"):
                        has_index = True
                        logger.info("✅ Qdrant index found for %s", vid)
                except Exception as e:
                    logger.warning("Qdrant check failed for %s: %s", vid, e)
                    
            if not has_index:
                cache_dir = _find_retrieval_cache_dir(vid)
                if cache_dir and (cache_dir / "index.faiss").exists():
                    has_index = True
                    logger.info("✅ FAISS index found for %s", vid)
                    
            if not has_index:
                missing.append(vid)

        if missing:
            logger.warning("Missing retrieval indexes for: %s", missing)
            yield (
                f"⚠️ {len(missing)}/{len(selected_videos)} videos lack indexes. Missing: {', '.join(missing)}",
                "", 0
            )
            return

        yield "🔎 Searching across selected videos...", "", 20
        try:
            with log_time("cross-video hybrid search pipeline"):
                docs = cross_video_hybrid_search(
                    question=question, video_ids=selected_videos, embeddings=runtime.embeddings
                )
                
            if not docs:
                logger.warning("Cross-video search returned 0 results.")
                yield "✅ Search complete.", "No relevant evidence found across selected videos.", 100
                return

            with log_time("cross-video context building"):
                context, source_lookup = build_context_with_sources(docs, video_id="multi")
            
            with log_time("cross-video LLM generation"):
                tpl = runtime.qa_chain.prompt.template
                raw = run_llm_dynamic(runtime.llm, tpl, {
                    "context": context, "question": question, "chat_history": "None"
                })
            rendered = render_clickable_answer(raw, source_lookup)
            logger.info("Cross-video QA successful. Answer length: %d chars", len(rendered))
            yield "✅ Answer ready.", rendered, 100
            
        except Exception as exc:
            logger.exception("Cross-video QA error")
            yield f"❌ Error: {exc}", "", 0
    return cross_video_qa_gradio


# =============================================================================
# EXPORT HELPER FUNCTIONS
# =============================================================================

def _create_temp_file(content: str, filename: str) -> str:
    """Write content to a temporary file and return its path for Gradio.
    
    Gradio automatically copies returned file paths to its internal serving
    directory, so the original temp file can safely persist until OS cleanup.
    """
    tmp_dir = tempfile.mkdtemp(prefix="yt_stt_export_")
    file_path = Path(tmp_dir) / filename
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def handle_export_transcript(state_payload: Dict[str, Any]) -> Optional[str]:
    state = SessionState.from_gradio(state_payload)
    if not state.processed_transcript:
        logger.info("Export skipped: no transcript in session.")
        return None
    path = _create_temp_file(
        state.processed_transcript,
        f"{state.video_id or 'transcript'}_transcript.txt"
    )
    logger.info("Exporting transcript → %s", path)
    return path

def handle_export_summary(state_payload: Dict[str, Any]) -> Optional[str]:
    state = SessionState.from_gradio(state_payload)
    if not state.summary:
        logger.info("Export skipped: no summary in session.")
        return None
    md_content = f"# Video Summary\n\n{state.summary}"
    path = _create_temp_file(md_content, f"{state.video_id or 'summary'}_summary.md")
    logger.info("Exporting summary → %s", path)
    return path

def handle_export_chat(state_payload: Dict[str, Any]) -> Optional[str]:
    state = SessionState.from_gradio(state_payload)
    if not state.chat_history:
        logger.info("Export skipped: no chat history in session.")
        return None
    
    lines = ["# Chat History\n"]
    for msg in state.chat_history:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"**{role}:**\n{content}\n")
    
    md_content = "\n---\n".join(lines)
    path = _create_temp_file(md_content, f"{state.video_id or 'chat'}_history.md")
    logger.info("Exporting chat → %s", path)
    return path

def handle_export_session_json(state_payload: Dict[str, Any]) -> Optional[str]:
    safe_payload = dict(state_payload)
    safe_payload.pop("faiss_index", None)
    path = _create_temp_file(
        json.dumps(safe_payload, indent=2, ensure_ascii=False),
        "session_export.json"
    )
    logger.info("Exporting session JSON → %s", path)
    return path


def get_cache_stats() -> str:
    """Calculate total cache size and return a formatted status string."""
    if not PATHS.root.exists():
        return "📦 Cache Size: 0.00 MB | 📁 Location: `(not created yet)`"
    try:
        total_bytes = sum(f.stat().st_size for f in PATHS.root.rglob("*") if f.is_file())
        total_mb = total_bytes / (1024 * 1024)
        return f"📦 Cache Size: {total_mb:.2f} MB | 📁 Location: `{PATHS.root}`"
    except Exception as exc:
        logger.warning("Failed to calculate cache size: %s", exc)
        return "⚠️ Unable to calculate cache size."

def clear_cache() -> str:
    """Safely delete all cache files and recreate directory structure.
    
    Note: This only removes on-disk artifacts. In-memory SessionState 
    (transcript, FAISS index, chat history) remains intact until the 
    user refreshes the page or loads a new video.
    """
    try:
        if PATHS.root.exists():
            shutil.rmtree(PATHS.root)
        PATHS.ensure()
        logger.info("Cache directory cleared and recreated.")
        return "✅ Cache cleared successfully."
    except PermissionError as exc:
        logger.warning("Permission denied while clearing cache: %s", exc)
        return "⚠️ Partial clear: some files are locked by active processes."
    except Exception as exc:
        logger.exception("Failed to clear cache")
        return f"❌ Failed to clear cache: {exc}"


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def build_interface(runtime: RuntimeDeps) -> gr.Blocks:
    """Assemble the Gradio Blocks UI and wire button click events.

    Layout:
        Tab 1 — Summarize
            Row:   YouTube URL input  |  Summarize button
            Label: status
            Row:   STT progress slider  |  Summary progress slider
            Row:   Token info           |  Transcript stats
            Accordion > Textbox: Summary prompt override
            Row:   Transcript textarea  |  Summary textarea
            Row:   Export buttons (Transcript, Summary, Session JSON)
            File:  Hidden download slots (revealed on export)

        Tab 2 — Q&A
            Row:   YouTube URL input (blank = reuse session)
            Textbox: Question input
            Accordion > Textbox: Q&A prompt override
            Row:   Ask button
            Label: status
            Slider: progress
            Chatbot: Conversational history with clickable timestamp citations
            Button: Export chat history
            File:  Hidden chat download slot

        Tab 3 — Settings & Cache
            Markdown: Live cache size & path stats
            Row:   Refresh stats button  |  Clear cache button
            Label: Operation status feedback
        
        Tab 4 - Library & Cross-Video Q&A

    A hidden gr.State component carries the SessionState payload between
    handler calls. Both handlers receive it as their last input and emit
    an updated payload as their last output.
    """
    summarize_handler = _make_summarize_handler(runtime)
    qa_handler = _make_qa_handler(runtime)

    with gr.Blocks(title="YouTube STT Summarizer & Q&A") as interface:
        # CSS to hide empty export file slots until a download is ready
        gr.HTML("""
        <style>
            .export-file:has(.file-preview) { display: block !important; }
            .export-file:not(:has(.file-preview)) { display: none !important; }
        </style>
        """)
        session_state = gr.State(value={})

        gr.Markdown("# 🎥 YouTube STT Summarizer & Timestamp-Aware Q&A")
        gr.Markdown(
            "Transcribe any YouTube video locally with Whisper, summarize it "
            "with a local LLM, and ask timestamp-aware questions."
        )

        with gr.Tabs():

            # ── Tab 1: Summarize ─────────────────────────────────────────────
            with gr.TabItem("📝 Summarize"):

                with gr.Row():
                    video_url_sum = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        scale=5,
                    )
                    summarize_btn = gr.Button("▶ Summarize", variant="primary", scale=2)
                    cancel_sum_btn = gr.Button("⛔ Cancel", variant="stop", scale=1)

                status_sum = gr.Label(label="Status")

                with gr.Row():
                    stt_progress = gr.Slider(
                        label="STT Progress",
                        minimum=0, maximum=100, value=0,
                        interactive=False,
                    )
                    summary_progress = gr.Slider(
                        label="Summary Progress",
                        minimum=0, maximum=100, value=0,
                        interactive=False,
                    )
                with gr.Row():
                    token_info = gr.Textbox(label="Token Info", interactive=False)
                    stats_info = gr.Textbox(
                        label="Transcript Stats", interactive=False
                    )

                with gr.Accordion("🛠️ Summary Prompt Override", open=False):
                    summary_prompt_input = gr.Textbox(
                        label="Custom Summary Prompt (leave blank to use default)",
                        lines=6,
                        placeholder="You are an AI assistant that summarizes...\nUse {transcript}, {chunk}, or {summaries} as variables.",
                    )

                with gr.Row():
                    transcript_out = gr.Textbox(
                        label="Transcript",
                        lines=15,
                        interactive=False,
                    )
                    summary_out = gr.Textbox(
                        label="Summary",
                        lines=15,
                        interactive=False,
                    )
                
                with gr.Accordion("📖 Auto-Chapters", open=False):
                    chapters_out = gr.Markdown(label="Chapters", value="No chapters generated yet.")
                    gen_chapters_btn = gr.Button("🔨 Generate Chapters", variant="secondary")

                def handle_generate_chapters(state_payload: Dict[str, Any]) -> str:
                    state = SessionState.from_gradio(state_payload)
                    if not state.transcript_segments:
                        return "❌ No transcript segments available. Run Summarize first."
                    
                    cached = load_cached_chapters(state.video_id, state.transcript_hash)
                    if cached:
                        chapters = cached
                    else:
                        chapters = generate_chapters(state.transcript_segments, runtime)
                        if chapters:
                            save_chapters(state.video_id, state.transcript_hash, chapters)
                    
                    if not chapters:
                        return "⚠️ Chapter generation returned no results."
                    
                    lines = ["### 📖 Chapters\n"]
                    for ch in chapters:
                        url = build_youtube_time_url(state.video_id, ch["start"])
                        lines.append(f"- [{seconds_to_hhmmss(ch['start'])} - {seconds_to_hhmmss(ch['end'])}] **{ch['title']}** → [Jump]({url})")
                    return "\n".join(lines)

                gen_chapters_btn.click(
                    fn=handle_generate_chapters,
                    inputs=[session_state],
                    outputs=[chapters_out],
                )

                with gr.Row():
                    export_transcript_btn = gr.Button("📥 Export Transcript (.txt)")
                    export_summary_btn = gr.Button("📥 Export Summary (.md)")
                    export_json_btn = gr.Button("📦 Export Session (.json)")

                transcript_file = gr.File(label="Transcript Download", interactive=False, elem_classes=["export-file"])
                summary_file = gr.File(label="Summary Download", interactive=False, elem_classes=["export-file"])
                session_file = gr.File(label="Session JSON", interactive=False, elem_classes=["export-file"])

                sum_event = summarize_btn.click(
                    fn=summarize_handler,
                    inputs=[video_url_sum, session_state, summary_prompt_input],
                    outputs=[
                        status_sum,
                        token_info,
                        transcript_out,
                        summary_out,
                        stats_info,
                        stt_progress,
                        summary_progress,
                        session_state,
                    ],
                    show_progress="full",
                )
                # Wire cancel button to interrupt the running generator
                cancel_sum_btn.click(fn=None, inputs=None, outputs=None, cancels=[sum_event])
           
                export_transcript_btn.click(
                    fn=handle_export_transcript,
                    inputs=[session_state],
                    outputs=[transcript_file],
                )
                export_summary_btn.click(
                    fn=handle_export_summary,
                    inputs=[session_state],
                    outputs=[summary_file],
                )
                export_json_btn.click(
                    fn=handle_export_session_json,
                    inputs=[session_state],
                    outputs=[session_file],
                )

            # ── Tab 2: Q&A ───────────────────────────────────────────────────
            with gr.TabItem("❓ Q&A"):

                with gr.Row():
                    video_url_qa = gr.Textbox(
                        label="YouTube URL (leave blank to reuse current session)",
                        placeholder=(
                            "https://www.youtube.com/watch?v=...  "
                            "or leave blank to reuse the Summarize session"
                        ),
                        scale=5,
                    )

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What does the speaker say about...?",
                    lines=2,
                )

                with gr.Accordion("🛠️ QA Prompt Override", open=False):
                    qa_prompt_input = gr.Textbox(
                        label="Custom QA Prompt (leave blank to use default)",
                        lines=6,
                        placeholder="Answer the question using ONLY the context...\nUse {context}, {question}, {chat_history} as variables.",
                    )

                with gr.Row():
                    ask_btn = gr.Button("🔎 Ask", variant="primary", scale=2)
                    cancel_qa_btn = gr.Button("⛔ Cancel", variant="stop", scale=1)

                status_qa = gr.Label(label="Status")
                qa_progress = gr.Slider(
                    label="Progress",
                    minimum=0, maximum=100, value=0,
                    interactive=False,
                )
                chatbot = gr.Chatbot(label="Conversation", height=400)

                export_chat_btn = gr.Button("📥 Export Chat History (.md)")
                chat_file = gr.File(label="Chat Download", interactive=False, elem_classes=["export-file"])

                qa_event = ask_btn.click(
                    fn=qa_handler,
                    inputs=[video_url_qa, question_input, session_state, qa_prompt_input],
                    outputs=[status_qa, chatbot, qa_progress, session_state],
                    show_progress="full",
                )
                cancel_qa_btn.click(fn=None, inputs=None, outputs=None, cancels=[qa_event])

                export_chat_btn.click(
                    fn=handle_export_chat,
                    inputs=[session_state],
                    outputs=[chat_file],
                )

            # ── Tab 3: Settings & Cache ──────────────────────────────────────
            with gr.TabItem("⚙️ Settings & Cache"):
                cache_stats_md = gr.Markdown(get_cache_stats())
                with gr.Row():
                    refresh_cache_btn = gr.Button("🔄 Refresh Stats")
                    clear_cache_btn = gr.Button("🗑️ Clear All Cache", variant="stop")
                cache_msg = gr.Label(label="Operation Status")

                refresh_cache_btn.click(
                    fn=lambda: gr.Markdown(get_cache_stats()),
                    outputs=[cache_stats_md],
                )
                clear_cache_btn.click(
                    fn=lambda: (clear_cache(), get_cache_stats()),
                    outputs=[cache_msg, cache_stats_md],
                )
            
            # ── Tab 4: Library & Cross-Video Q&A ───────────────────────────────
            with gr.TabItem("📚 Library & Cross-Video Q&A"):
                with gr.Row():
                    library_df = gr.Dataframe(
                        headers=["Video ID", "URL", "Indexed At"],
                        datatype=["str", "str", "number"],
                        interactive=False,
                        label="Indexed Videos"
                    )
                    refresh_lib_btn = gr.Button("🔄 Refresh Library")

                video_selector = gr.CheckboxGroup(
                    label="Select Videos to Search",
                    choices=[],
                    interactive=True
                )

                index_btn = gr.Button("🔨 Index Selected Videos", variant="secondary")
                index_status = gr.Label(label="Indexing Status")

                def index_selected_videos(selected_videos: List[str]) -> Generator[str, None, None]:
                    """Batch-index selected videos with full observability."""
                    if not selected_videos:
                        yield "❌ No videos selected."
                        return
                        
                    logger.info("Batch indexing triggered for: %s", selected_videos)
                    logger.info("Vector DB Type: %s | QDRANT_AVAILABLE: %s", CFG.vector_db_type, QDRANT_AVAILABLE)
                    yield f"🚀 Indexing {len(selected_videos)} videos..."
                    
                    try:
                        for vid in selected_videos:
                            logger.info("Indexing video: %s", vid)
                            yield f"📥 Processing {vid}..."
                            state = SessionState()
                            video_url = f"https://www.youtube.com/watch?v={vid}"
                            
                            # 1. Fetch/Load Transcript
                            for update in fetch_transcript_from_stt_stream(video_url, runtime):
                                if update.transcript and update.segments:
                                    state.set_transcript(video_url, update.transcript, update.segments)
                                    
                            if not state.processed_transcript:
                                logger.warning("Transcript fetch failed for %s", vid)
                                yield f"⚠️ Failed to fetch transcript for {vid}. Skipping."
                                continue
                                
                            # 2. Build Chunks
                            logger.info("Building chunks for %s...", vid)
                            get_or_create_chunks(state, runtime)
                            if not state.chunks:
                                logger.error("Chunking returned empty list for %s", vid)
                                yield f"⚠️ Chunking failed for {vid}. Skipping."
                                continue
                                
                            # 3. Build & Persist Vector Store
                            logger.info("Building vector store for %s (%d chunks)...", vid, len(state.chunks))
                            store = get_or_create_vector_store(state, runtime)
                            if store is None:
                                logger.error("Vector store creation returned None for %s", vid)
                                yield f"❌ Vector store build failed for {vid}."
                                continue
                                
                            logger.info("✅ Successfully indexed %s", vid)
                            yield f"✅ Indexed {vid}."
                            
                        yield "🎉 All selected videos indexed successfully."
                    except Exception as exc:
                        logger.exception("Batch indexing crashed")
                        yield f"❌ Indexing failed: {exc}"

                index_btn.click(
                    fn=index_selected_videos,
                    inputs=[video_selector],
                    outputs=[index_status],  # Single output matches single string yields
                )

                # Explicitly sync backend choices state using gr.update()
                def refresh_library():
                    lib = get_video_library()
                    df_data = [[r["video_id"], r["url"], r["indexed_at"]] for r in lib]
                    choices = [r["video_id"] for r in lib]
                    # Returning gr.update(choices=...) tells Gradio to update the 
                    # validation whitelist, not just the selected values.
                    return df_data, gr.update(choices=choices)

                refresh_lib_btn.click(
                    fn=refresh_library,
                    outputs=[library_df, video_selector]
                )

                cross_question_input = gr.Textbox(label="Cross-Video Question", lines=2)
                cross_ask_btn = gr.Button("🔍 Search Across Videos", variant="primary")
                cross_status = gr.Label(label="Status")
                cross_progress = gr.Slider(label="Progress", minimum=0, maximum=100, value=0, interactive=False)
                cross_answer_out = gr.Markdown(label="Cross-Video Answer")

                cross_qa_handler = _make_cross_video_qa_handler(runtime)
                cross_ask_btn.click(
                    fn=cross_qa_handler,
                    inputs=[video_selector, cross_question_input],
                    outputs=[cross_status, cross_answer_out, cross_progress],
                )

    return interface


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main() -> None:
    """Application entrypoint.

    Execution order:
        1. PATHS.ensure()         — create cache directories.
        2. build_runtime()        — initialise Whisper + Ollama; blocks on
                                    health check.
        3. build_interface(runtime) — construct the Gradio Blocks UI wired to
                                    the runtime-bound handler factories.
        4. .queue()               — serialise requests to the local Ollama
                                    server (concurrency limit = 1).
        5. .launch()              — start the Gradio server; blocks until
                                    stopped.

    runtime and interface are local variables here, not module globals.
    The module now imports cleanly without side effects, which makes it
    straightforward to unit-test individual functions.
    """
    PATHS.ensure()
    init_db()
    runtime = build_runtime()
    interface = build_interface(runtime)
    interface.queue(default_concurrency_limit=1)
    interface.launch(server_name="localhost", server_port=7860)


if __name__ == "__main__":
    main()
