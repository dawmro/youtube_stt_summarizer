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



import hashlib
import json
import logging
import os
import re
import requests
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, NamedTuple, Optional, Tuple

import tiktoken
from faster_whisper import WhisperModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


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
TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


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
        "embed_chunk_size": CFG.embed_chunk_size,
        "embed_chunk_overlap_segments": CFG.embed_chunk_overlap_segments,
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

Context:
{context}

Question: {question}

Answer:""",
        ["context", "question"],
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
    )


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def transcribe_audio(
    audio_path: Path, whisper_model: WhisperModel
) -> Tuple[str, List[TranscriptSegment]]:
    """Run faster-whisper and return transcript text plus typed segments.

    Each Whisper segment becomes a TranscriptSegment carrying its start/end
    times.  The plain-text transcript is the segments joined by newlines.
    """
    kwargs: Dict[str, Any] = {
        "beam_size": CFG.whisper_beam_size,
        "vad_filter": CFG.whisper_vad_filter,
        "condition_on_previous_text": CFG.whisper_condition_on_previous_text,
    }
    if CFG.whisper_language:
        kwargs["language"] = CFG.whisper_language

    with log_time("whisper transcription"):
        segments, info = whisper_model.transcribe(str(audio_path), **kwargs)

    transcript_lines: List[str] = []
    structured_segments: List[TranscriptSegment] = []

    for seg_idx, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue

        transcript_lines.append(text)
        structured_segments.append(
            TranscriptSegment(
                segment_id=seg_idx,
                start=float(seg.start) if seg.start is not None else 0.0,
                end=float(seg.end) if seg.end is not None else 0.0,
                text=text,
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
# TRANSCRIPT CACHE
# =============================================================================

def transcript_record_path(video_id: str) -> Path:
    """Derive the cache file path for a transcript.

    The STT_CONFIG_HASH is embedded in the filename so any change to Whisper
    settings automatically routes to a new file — no stale cache reads.

    Pattern: <video_id>__<stt_config_hash>.json
    """
    return PATHS.transcript / f"{video_id}__{STT_CONFIG_HASH}.json"


def save_transcript(
    video_id: str, transcript: str, segments: List[TranscriptSegment]
) -> None:
    """Persist transcript text and timestamped segments together.

    Both are stored in one record so the summary (which needs text) and the
    retrieval pipeline (which needs segment timestamps) always stay in sync.
    """
    write_json_atomic(
        transcript_record_path(video_id),
        {
            "video_id": video_id,
            "transcript": transcript,
            "transcript_hash": text_hash(transcript),
            "segments": [seg.to_dict() for seg in segments],
            "transcript_schema_version": CFG.transcript_schema_version,
            "stt_config_hash": STT_CONFIG_HASH,
            "stt_config": current_stt_config(),
            "saved_at": time.time(),
        },
    )


def load_cached_transcript(
    video_id: str,
) -> Optional[Tuple[str, List[TranscriptSegment], str]]:
    """Load a transcript from cache when the STT config hash matches.

    The config hash is already encoded in the filename, so a file that exists
    at the derived path is guaranteed to match the current settings.  We only
    validate that the payload is structurally complete (non-empty transcript
    and at least one segment).

    Returns (transcript_text, segments, transcript_hash) or None on miss.
    """
    data = read_json(transcript_record_path(video_id))
    if not data:
        return None

    transcript = str(data.get("transcript", "")).strip()
    raw_segments = data.get("segments")
    if not transcript or not isinstance(raw_segments, list) or not raw_segments:
        return None

    segments = [TranscriptSegment.from_dict(item) for item in raw_segments]
    transcript_hash_value = data.get("transcript_hash") or text_hash(transcript)
    return transcript, segments, transcript_hash_value


# =============================================================================
# TYPED GENERATOR YIELDS
# ============================================================================

class SummaryUpdate(NamedTuple):
    """Streaming update yielded by summarize_transcript_stream.

    Fields:
        message  — human-readable status line shown in the UI status label.
        summary  — the finished summary text, or None while still generating.
        progress — integer 0-100 for the progress bar.
    """
    message: str
    summary: Optional[str]
    progress: int


# =============================================================================
# SUMMARIZATION PIPELINE
# =============================================================================

def summarize_transcript_stream(
    text: str,
    runtime: RuntimeDeps,
) -> Generator[SummaryUpdate, None, None]:
    """Summarize a transcript, yielding typed status updates for the UI.

    For transcripts that fit within the model context window the full text is
    sent to the LLM in a single call (direct mode).  
    """
    if estimate_tokens(text) <= CFG.max_transcript_tokens:
        yield SummaryUpdate("📝 Generating direct summary...", None, 30)
        with log_time("direct summary generation"):
            summary = runtime.summary_chain.predict(transcript=text).strip()
        if not summary:
            raise RuntimeError("Direct summary generation returned empty output.")
        yield SummaryUpdate("✅ Final summary ready.", summary, 100)
        return

    # Transcripts longer than the context window are not yet supported.
    raise RuntimeError(
        f"Transcript is too long ({estimate_tokens(text)} tokens) for direct "
        "summarization. Chunked support coming soon."
    )



video_url = "https://www.youtube.com/watch?v=BSuAgw8Lc1Y"
video_id = require_video_id(video_url)

source_dir = PATHS.ytdlp / video_id
audio_dir = PATHS.audio / video_id
wav_path = audio_dir / "audio.wav"

with log_time("Getting source audio"):
    source_audio = download_audio(video_id, source_dir)
with log_time("Converting to WAV"):    
    convert_to_wav_16k_mono(source_audio, wav_path)
logger.info(build_youtube_time_url(video_id, 245))

logger.info(STT_CONFIG_HASH)
logger.info(SUMMARY_CONFIG_HASH)
logger.info(RETRIEVAL_CONFIG_HASH)

whisper_model = WhisperModel(
    CFG.whisper_model_size,
    device=CFG.whisper_device,
    compute_type=CFG.whisper_compute_type,
)

with log_time("Transcribing audio"): 
    transcript, segments = transcribe_audio(wav_path, whisper_model)
logger.info(f"Transcript: \n{transcript}")
logger.info(f"Structured Segments: \n{segments}")

save_transcript(video_id, transcript, segments)
cached = load_cached_transcript(video_id)
transcript, segments, transcript_hash_value = cached
logger.info(f"Cached transcript: \n{transcript}")
logger.info(f"Cached structured segments: \n{segments}")
logger.info(f"Transcript hash value: {transcript_hash_value}")

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

tokens = estimate_tokens(transcript)
logger.info(f"Estimated tokens: {tokens}")

runtime = build_runtime()
message, summary = summarize_transcript_stream(transcript, runtime)

logger.info(message, summary)