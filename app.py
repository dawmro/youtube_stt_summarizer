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

import gradio as gr
import tiktoken
from faster_whisper import WhisperModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
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

# Compiled patterns used by render_clickable_answer.
# Defined at module level so they are compiled once, not per call.
SOURCE_GROUP_PATTERN = re.compile(r"\[((?:S\d+\s*(?:,\s*S\d+\s*)*))\]")
SOURCE_SINGLE_PATTERN = re.compile(r"(?<!\[)(S\d+)(?!\])")


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
# TRANSCRIPT CACHE AND PIPELINE
# =============================================================================

def transcript_record_path(video_id: str) -> Path:
    return PATHS.transcript / f"{video_id}__{STT_CONFIG_HASH}.json"


def load_cached_transcript(
    video_id: str,
) -> Optional[Tuple[str, List[TranscriptSegment], str]]:
    """Load transcript cache when the STT config hash (encoded in filename) matches."""
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


def save_transcript(
    video_id: str, transcript: str, segments: List[TranscriptSegment]
) -> None:
    """Persist transcript text and timestamped segments together."""
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

    with log_time("whisper transcription"):
        segments, info = runtime.whisper.transcribe(str(audio_path), **kwargs)

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
    audio_dir = PATHS.audio / video_id
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
    wav_path = audio_dir / "audio.wav"
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
            summary = runtime.summary_chain.predict(transcript=text).strip()
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
                chunk_summary = runtime.chunk_summary_chain.predict(
                    chunk=chunk
                ).strip()
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
                final_summary = runtime.reduce_summary_chain.predict(
                    summaries=merged
                ).strip()
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

def summary_record_path(
    video_id: str, transcript_hash_value: str, mode: str
) -> Path:
    """Derive the cache file path for a summary.

    Three inputs are encoded in the filename:
        - video_id            — identifies the video.
        - mode                — "direct" or "chunked"; kept separate so a
                                change in strategy forces a fresh generation.
        - SUMMARY_CONFIG_HASH — invalidates the cache when the LLM model,
                                context limits, or prompt version change.
        - transcript_hash_value — invalidates the cache when the transcript
                                  content changes (e.g. different Whisper run).

    Pattern: <video_id>__<mode>__<summary_config_hash>__<transcript_hash>.json
    """
    return (
        PATHS.summary
        / f"{video_id}__{mode}__{SUMMARY_CONFIG_HASH}__{transcript_hash_value}.json"
    )


def save_summary(
    video_id: str, transcript_hash_value: str, mode: str, summary: str
) -> None:
    """Persist summary output tied to transcript content and summary strategy.

    The mode field ("direct" / "chunked") is stored in the payload for
    observability — it lets you tell at a glance which strategy produced the
    cached result without reading the content.
    """
    write_json_atomic(
        summary_record_path(video_id, transcript_hash_value, mode),
        {
            "video_id": video_id,
            "transcript_hash": transcript_hash_value,
            "summary": summary,
            "mode": mode,
            "summary_config_hash": SUMMARY_CONFIG_HASH,
            "summary_config": current_summary_config(),
            "saved_at": time.time(),
        },
    )


def load_cached_summary(
    video_id: str, transcript_hash_value: str, mode: str
) -> Optional[str]:
    """Load summary cache when the filename (config + transcript hash) matches.

    No redundant in-body hash checks: the filename already encodes the full
    cache key so a file that exists at the derived path is guaranteed to match
    the current settings and transcript content.

    Returns the summary string or None on a cache miss.
    """
    data = read_json(summary_record_path(video_id, transcript_hash_value, mode))
    if not data:
        return None
    summary = str(data.get("summary", "")).strip()
    return summary or None


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalChunk":
        return cls(
            chunk_id=int(data["chunk_id"]),
            text=str(data["text"]),
            start=float(data["start"]),
            end=float(data["end"]),
            segment_ids=[int(x) for x in data.get("segment_ids", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

# =============================================================================
# RETRIEVAL CHUNKING
# =============================================================================

def build_retrieval_chunks(segments: List[TranscriptSegment]) -> List[RetrievalChunk]:
    """Group TranscriptSegment objects into RetrievalChunk instances.

    Chunking strategy:
    - Accumulate segments until the combined text would exceed embed_chunk_size
      characters (CFG.embed_chunk_size).
    - After emitting a chunk, step back by embed_chunk_overlap_segments so the
      next chunk starts overlap-many segments before the current boundary.
      This prevents context from being silently cut at chunk edges.
    - Segments whose text is empty or whitespace-only are skipped entirely.
    - chunk_id values are assigned sequentially starting from 0.

    Returns an empty list when all input segments have empty text.
    """
    # Filter out segments with no usable text upfront.
    valid = [s for s in segments if s.text.strip()]
    if not valid:
        return []

    chunks: List[RetrievalChunk] = []
    chunk_id = 0
    idx = 0

    while idx < len(valid):
        window: List[TranscriptSegment] = []
        char_count = 0

        # Grow the window until the next segment would push us over the budget.
        j = idx
        while j < len(valid):
            seg = valid[j]
            seg_len = len(seg.text)
            if window and char_count + seg_len > CFG.embed_chunk_size:
                break
            window.append(seg)
            char_count += seg_len
            j += 1

        if not window:
            # Safety: single segment longer than the budget — include it alone.
            window = [valid[idx]]
            j = idx + 1

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

        # Slide the window forward, stepping back by the overlap amount so the
        # next chunk re-uses the last overlap-many segments of this one.
        advance = max(1, len(window) - CFG.embed_chunk_overlap_segments)
        idx += advance

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
        last_question — the last question the user asked.
        last_answer   — the rendered Markdown answer to last_question.
        chunks        — List[RetrievalChunk] built from transcript_segments.
        faiss_index   — FAISS vector store built from chunks.
    """
    # --- transcript fields ---
    video_url: str = ""
    video_id: str = ""
    processed_transcript: str = ""
    transcript_hash: str = ""
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)

    # --- derived fields (cleared when transcript changes) ---
    summary: str = ""
    last_question: str = ""
    last_answer: str = ""
    chunks: Optional[List[RetrievalChunk]] = None
    faiss_index: Optional[Any] = None

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
        self.last_question = ""
        self.last_answer = ""
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
            "last_question": self.last_question,
            "last_answer": self.last_answer,
            "chunks": (
                [c.to_dict() for c in self.chunks]
                if self.chunks is not None
                else None
            ),
            # Kept by reference — not serialised to JSON.
            "faiss_index": self.faiss_index,
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
            last_question=str(payload.get("last_question", "")),
            last_answer=str(payload.get("last_answer", "")),
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


def get_or_create_chunks(state: SessionState) -> None:
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
        state.chunks = build_retrieval_chunks(state.transcript_segments)
    logger.info("Built %d retrieval chunks.", len(state.chunks))


def get_or_create_faiss(state: SessionState, runtime: RuntimeDeps) -> FAISS:
    """Return the FAISS index for the active session, building it if necessary.

    Three-layer lookup:
        1. In-memory  — state.faiss_index already set from this session.
        2. Disk cache — load_retrieval_cache finds matching files on disk.
        3. Build fresh — embed state.chunks, build the index, persist to disk.

    Callers must call get_or_create_chunks(state) before this function so
    state.chunks is populated for layers 2 and 3.
    """
    if state.faiss_index is not None:
        logger.info("FAISS index found in session memory — reusing.")
        return state.faiss_index

    cached = load_retrieval_cache(
        state.video_id, state.transcript_hash, runtime.embeddings
    )
    if cached:
        state.chunks, state.faiss_index = cached
        return state.faiss_index

    if not state.chunks:
        raise RuntimeError(
            "Cannot build FAISS index: state.chunks is empty. "
            "Call get_or_create_chunks(state) first."
        )

    logger.info("Building FAISS index from %d chunks...", len(state.chunks))
    texts = [chunk.text for chunk in state.chunks]
    metadatas = [
        {"start": chunk.start, "end": chunk.end, "chunk_id": chunk.chunk_id}
        for chunk in state.chunks
    ]

    with log_time("FAISS index build"):
        faiss_index = FAISS.from_texts(texts, runtime.embeddings, metadatas=metadatas)

    state.faiss_index = faiss_index
    save_retrieval_cache(
        state.video_id, state.transcript_hash, state.chunks, faiss_index
    )
    logger.info("FAISS index built and persisted to retrieval cache.")
    return faiss_index

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
        url = build_youtube_time_url(video_id, start)

        source_lookup[label] = SourceRef(
            start=start,
            end=end,
            label=time_label,
            url=url,
            chunk_id=chunk_id,
            text=text,
        )
        context_parts.append(f"[{label}] {text}")

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
# GRADIO HANDLERS
# =============================================================================

def _make_summarize_handler(runtime: RuntimeDeps):
    """Factory returning summarize_video_gradio closed over runtime.

    Moving the runtime dependency into a closure instead of referencing a
    module-level singleton means the module can be imported in tests without
    triggering Whisper and Ollama initialisation.
    """
    def summarize_video_gradio(
        video_url: str,
        state_payload: Dict[str, Any],
    ) -> Generator:
        """Run transcript + summary pipeline and stream UI updates.

        Drives the full transcript → summary pipeline, updating seven Gradio
        outputs at each yield step:
            (status, token_info, transcript, summary, stats,
             stt_progress, summary_progress, state)
        """
        state = SessionState.from_gradio(state_payload)
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
                state.processed_transcript, runtime
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

        except Exception as exc:
            logger.exception("Summarize pipeline error")
            yield (
                f"❌ Error: {exc}",
                "", "", "", "", 0, 0, state.to_gradio(),
            )

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
    ) -> Generator:
        """Run timestamp-aware transcript Q&A and stream UI updates.

        Pipeline: validate input → refresh transcript if needed → build chunks
        → build FAISS index → similarity search → build context
        → generate answer → render citations.

        Streams four outputs: (status, answer, progress, state).
        Skips transcript fetch if the session already has it cached.
        """
        state = SessionState.from_gradio(state_payload)
        question = user_question.strip()

        # ── Input validation ─────────────────────────────────────────────────
        if not question:
            yield "❌ Please provide a question.", "", 0, state.to_gradio()
            return
        if video_url.strip() and not get_video_id(video_url):
            yield "❌ Invalid YouTube URL.", "", 0, state.to_gradio()
            return
        if not video_url.strip() and not state.processed_transcript:
            yield (
                "❌ No transcript in session. Provide a YouTube URL first.",
                "", 0, state.to_gradio(),
            )
            return

        try:
            progress = 0
            yield "🚀 Starting Q&A pipeline...", "", progress, state.to_gradio()

            # ── Transcript fetch (skipped when session already has it) ────────
            if state.needs_transcript_refresh(video_url):
                transcript: Optional[str] = None
                segments: Optional[List[TranscriptSegment]] = None

                for update in fetch_transcript_from_stt_stream(video_url, runtime):
                    # Scale STT progress into 0-40 % of the overall bar.
                    progress = min(40, max(5, int(update.progress * 0.4)))
                    transcript = update.transcript or transcript
                    segments = update.segments or segments
                    yield update.message, "", progress, state.to_gradio()

                if not transcript or not segments:
                    raise RuntimeError("Failed to fetch transcript.")
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
            get_or_create_chunks(state)

            # ── FAISS index ──────────────────────────────────────────────────
            progress = 65
            yield (
                "🗂️ Loading or building FAISS index...",
                "", progress, state.to_gradio(),
            )
            faiss_index = get_or_create_faiss(state, runtime)

            # ── Similarity search ────────────────────────────────────────────
            progress = 80
            yield (
                "🔎 Searching transcript for relevant context...",
                "", progress, state.to_gradio(),
            )
            with log_time("FAISS similarity search"):
                docs = faiss_index.similarity_search(question, k=CFG.retrieval_top_k)

            if not docs:
                state.last_question = question
                state.last_answer = (
                    "I couldn't find relevant transcript evidence "
                    "for that question."
                )
                yield "✅ Answer ready.", state.last_answer, 100, state.to_gradio()
                return

            context, source_lookup = build_context_with_sources(docs, state.video_id)
            if not context.strip():
                state.last_question = question
                state.last_answer = (
                    "I couldn't build usable context from the "
                    "retrieved transcript chunks."
                )
                yield "✅ Answer ready.", state.last_answer, 100, state.to_gradio()
                return

            # ── LLM answer generation ────────────────────────────────────────
            progress = 90
            yield (
                f"🧠 Context ready "
                f"({estimate_tokens(context)} tokens). Generating answer...",
                "", progress, state.to_gradio(),
            )
            with log_time("QA generation"):
                raw_answer = runtime.qa_chain.predict(
                    context=context, question=question
                ).strip()

            state.last_question = question
            state.last_answer = render_clickable_answer(raw_answer, source_lookup)
            yield "✅ Answer ready.", state.last_answer, 100, state.to_gradio()

        except Exception as exc:
            logger.exception("Q&A pipeline error")
            yield f"❌ Error: {exc}", "", 0, state.to_gradio()

    return answer_question_gradio


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
            Text:  token info
            Text:  transcript stats
            Row:   Transcript textarea  |  Summary textarea

        Tab 2 — Q&A
            Row:   YouTube URL input (blank = reuse session)
            Text:  question input
            Row:   Ask button
            Label: status
            Slider: progress
            Markdown: answer (supports clickable timestamp links)

    A hidden gr.State component carries the SessionState payload between
    handler calls.  Both handlers receive it as their last input and emit
    an updated payload as their last output.
    """
    summarize_handler = _make_summarize_handler(runtime)
    qa_handler = _make_qa_handler(runtime)

    with gr.Blocks(title="YouTube STT Summarizer & Q&A") as interface:
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
                    summarize_btn = gr.Button(
                        "▶ Summarize", variant="primary", scale=1
                    )

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

                token_info = gr.Textbox(label="Token Info", interactive=False)
                stats_info = gr.Textbox(
                    label="Transcript Stats", interactive=False
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

                summarize_btn.click(
                    fn=summarize_handler,
                    inputs=[video_url_sum, session_state],
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

                with gr.Row():
                    ask_btn = gr.Button("🔎 Ask", variant="primary")

                status_qa = gr.Label(label="Status")
                qa_progress = gr.Slider(
                    label="Progress",
                    minimum=0, maximum=100, value=0,
                    interactive=False,
                )
                answer_out = gr.Markdown(label="Answer")

                ask_btn.click(
                    fn=qa_handler,
                    inputs=[video_url_qa, question_input, session_state],
                    outputs=[status_qa, answer_out, qa_progress, session_state],
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
    runtime = build_runtime()
    interface = build_interface(runtime)
    interface.queue(default_concurrency_limit=1)
    interface.launch(server_name="localhost", server_port=7860)


if __name__ == "__main__":
    main()
