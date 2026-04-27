"""
[Proof of Concept] YouTube STT Summarizer & Q&A Tool

Business goal:
- Download audio from a YouTube video.
- Transcribe it locally with faster-whisper.
- Cache transcript and retrieval artifacts so repeated runs stay fast.
- Summarize the transcript with Ollama.
- Answer questions from the transcript with FAISS retrieval.
"""



import os
# fix for two different OpenMP implementations loaded at the same time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2) 

import gradio as gr
import re
import logging
import subprocess
import tiktoken
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, List, Tuple

from faster_whisper import WhisperModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from pathlib import Path

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / "cache_poc"
YTDLP_CACHE_DIR = CACHE_DIR / "yt_dlp_cache"
WHISPER_CACHE_DIR = CACHE_DIR / "faster_whisper_cache"
TRANSCRIPT_CACHE_DIR = CACHE_DIR / "transcript_cache"

LLM_MODEL = "llama3.1:8b-instruct-q8_0"
EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_BASE_URL = "http://localhost:11434"

LLM_CONTEXT_LIMIT = 8192   # set this to your llama3.1 context window
SAFETY_MARGIN = 1024        # prompt + formatting overhead
MAX_TRANSCRIPT_TOKENS = LLM_CONTEXT_LIMIT - SAFETY_MARGIN

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# faster-whisper settings
WHISPER_MODEL_SIZE = "small"   # can be "large-v3", "medium", "small", etc.
WHISPER_DEVICE = "cpu"            # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = "int8"     # "float16" for cuda, "int8" for cpu


# =============================================================================
# GLOBAL INITIALIZATION
# =============================================================================

logger.info("Initializing models at startup...")

GLOBAL_LLM = Ollama(
    model=LLM_MODEL,
    temperature=0.7,
    top_p=0.9,
    base_url=OLLAMA_BASE_URL
)

GLOBAL_EMBEDDINGS = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

GLOBAL_WHISPER = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

logger.info("✅ Models initialized successfully")

# =============================================================================
# PROMPTS
# =============================================================================

SUMMARY_CHAIN = LLMChain(
    llm=GLOBAL_LLM,
    prompt=PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that summarizes transcripts of YouTube videos.

Instructions:
- Write a concise but informative summary.
- Focus only on what is actually said.
- Ignore timestamps and filler words.
- Capture all details, numbers and values that appear.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Transcript:
{transcript}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Summary:""",
        input_variables=["transcript"]
    )
)


CHUNK_SUMMARY_CHAIN = LLMChain(
    llm=GLOBAL_LLM,
    prompt=PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that summarizes transcripts of parts of YouTube videos.

Instructions:
- Write a detailed and informative summary.
- Focus only on what is actually said.
- Ignore timestamps and filler words.
- Capture all details, numbers and values that appear.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Transcript chunk:
{chunk}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Chunk summary:""",
        input_variables=["chunk"]
    )
)


QA_CHAIN = LLMChain(
    llm=GLOBAL_LLM,
    prompt=PromptTemplate(
        template="""Answer the question using ONLY the context.

If the answer is not in the context, respond:
"I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
)

# =============================================================================
# UTILS
# =============================================================================

@contextmanager
def log_time(label: str):
    start = time.perf_counter()
    logger.info("START: %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("END: %s (%.2fs)", label, elapsed)


def estimate_tokens(text: str) -> int:
    return len(TIKTOKEN_ENC.encode(text))


def transcript_path(video_id: str) -> Path:
    return TRANSCRIPT_CACHE_DIR / f"{video_id}.txt"


def load_cached_transcript(video_id: str) -> Optional[str]:
    path = transcript_path(video_id)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def save_transcript(video_id: str, transcript: str) -> None:
    path = transcript_path(video_id)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(transcript, encoding="utf-8")
    tmp_path.replace(path)


def ensure_dirs():
    YTDLP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=128)
def get_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.
    """
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"shorts\/([a-zA-Z0-9_-]{11})"
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None


def run_command(cmd: List[str]) -> None:
    try:
        logger.info("Running command: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Command failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

    except FileNotFoundError:
        raise RuntimeError(f"Executable not found: {cmd[0]}. Is it installed and in PATH?")


def download_audio_ytdlp(video_url: str, output_dir: str) -> str:
    """
    Download best audio using yt-dlp.
    Returns path to downloaded audio file.
    """
    out_template = os.path.join(output_dir, "audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", out_template,
        video_url
    ]

    run_command(cmd)

    # Find downloaded mp3
    for f in os.listdir(output_dir):
        if f.startswith("audio.") and f.endswith(".mp3"):
            return os.path.join(output_dir, f)

    raise FileNotFoundError("yt-dlp did not produce an audio file.")


def convert_to_wav_16k_mono(input_audio: str, output_wav: str) -> str:
    """
    Convert audio into whisper-friendly WAV: 16kHz mono PCM.
    """
    # ensure output directory exists
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_audio,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_wav
    ]

    run_command(cmd)
    return output_wav


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using faster-whisper.
    Returns full transcript text.
    """
    logger.info("Transcribing audio with faster-whisper...")

    segments, info = GLOBAL_WHISPER.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=1000,
            speech_pad_ms=200,
        ),
        condition_on_previous_text=True,
    )
    

    lines = []
    for seg in segments:
        # seg.start, seg.end available if you want timestamps
        text = seg.text.strip()
        if text:
            lines.append(text)

    transcript = "\n".join(lines).strip()
    if not transcript:
        raise RuntimeError("STT returned an empty transcript.")

    logger.info(
        "Transcription done (language=%s, prob=%.3f, chars=%d)",
        getattr(info, "language", "unknown"),
        getattr(info, "language_probability", 0.0),
        len(transcript),
    )
    return transcript


def chunk_transcript_for_embeddings(transcript: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_text(transcript)


def chunk_transcript_for_summary(
    transcript: str,
    target_tokens: int | None = None,
    overlap_tokens: int = 256,
) -> List[str]:
    if target_tokens is None:
        target_tokens = MAX_TRANSCRIPT_TOKENS // 2

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", " ", ""],
        length_function=estimate_tokens,
    )
    return splitter.split_text(transcript)


def retrieve_context(question: str, faiss_index: FAISS, k: int = 4) -> str:
    docs = faiss_index.similarity_search(question, k=k)
    return "\n".join(d.page_content for d in docs)


def fetch_transcript_from_stt(video_url: str) -> str:
    """
    Full pipeline:
    URL -> download audio -> convert -> STT -> transcript
    """
    
    # Using video_id as cache key is safe unless two users request the same video simultaneously.
    # TODO: Possible fix to avoid corruption: write transcript to a temp file then rename atomically.
    video_id = get_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")
    
    logger.info("Processing video_id=%s", video_id)
    
    # 1) check transcript cache first
    cached = load_cached_transcript(video_id)
    if cached:
        logger.info("Using cached transcript for video_id=%s", video_id)
        return cached
    
    # 2) download mp3 into stable folder
    video_ytdlp_dir = YTDLP_CACHE_DIR / video_id
    video_whisper_dir = WHISPER_CACHE_DIR / video_id

    video_ytdlp_dir.mkdir(parents=True, exist_ok=True)
    video_whisper_dir.mkdir(parents=True, exist_ok=True)

    with log_time("yt-dlp download"):
        mp3_path = download_audio_ytdlp(video_url, str(video_ytdlp_dir))

    wav_path = str(video_whisper_dir / "audio.wav")
    with log_time("ffmpeg convert mp3->wav"):
        convert_to_wav_16k_mono(mp3_path, wav_path)

    with log_time("whisper transcription"):
        transcript = transcribe_audio(wav_path)

    if not transcript.strip():
        raise RuntimeError("STT returned an empty transcript.")
    
    logger.info("Transcript length: %d chars", len(transcript))
    
    # 3) save transcript permanently
    save_transcript(video_id, transcript)

    logger.info("Transcript cached at %s", transcript_path(video_id))
    return transcript


def hierarchical_summarize(transcript: str) -> str:
    chunks = chunk_transcript_for_summary(
        transcript=transcript,
        target_tokens=MAX_TRANSCRIPT_TOKENS // 2,
        overlap_tokens=256,
    )
    logger.info("Transcript split into %d chunks", len(chunks))

    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        logger.info("Summarizing chunk %d/%d...", i, len(chunks))
        with log_time(f"chunk summary {i}/{len(chunks)}"):
            chunk_summary = CHUNK_SUMMARY_CHAIN.run({"chunk": chunk})
        chunk_summaries.append(chunk_summary.strip())

    combined_summary_text = "\n\n".join(chunk_summaries)
    combined_tokens = estimate_tokens(combined_summary_text)
    if combined_tokens > MAX_TRANSCRIPT_TOKENS:
        logger.warning("Combined chunk summaries still too long (%d tokens). Re-summarizing...", combined_tokens)
        return hierarchical_summarize(combined_summary_text)

    logger.info("Generating final summary from %d chunk summaries...", len(chunk_summaries))
    with log_time("combined summary generation"):
        final_summary = SUMMARY_CHAIN.run({"transcript": combined_summary_text})

    return final_summary

# =============================================================================
# GRADIO FUNCTIONS
# =============================================================================

def summarize_video_gradio(video_url: str, state_data: dict) -> Tuple[str, str, str, dict]:
    if not video_url:
        return "Please provide a valid YouTube URL.", "", "", state_data

    try:
        transcript = fetch_transcript_from_stt(video_url)

        token_count = estimate_tokens(transcript)
        mode = "chunked" if token_count > MAX_TRANSCRIPT_TOKENS else "direct"
        token_info = f"{token_count} tokens (limit ~{MAX_TRANSCRIPT_TOKENS}) [mode={mode}]"
        chunks = chunk_transcript_for_embeddings(transcript)

        state_data["video_url"] = video_url
        state_data["processed_transcript"] = transcript
        state_data["chunks"] = chunks
        state_data.pop("faiss_index", None)

        if token_count > MAX_TRANSCRIPT_TOKENS:
            logger.warning("Transcript too long (%d tokens). Using hierarchical summarization.", token_count)
            summary = hierarchical_summarize(transcript)
        else:
            with log_time("summary generation"):
                summary = SUMMARY_CHAIN.run({"transcript": transcript})

        return transcript, summary, token_info, state_data

    except Exception as e:
        logger.error("Summarize error: %s", e)
        return f"Error: {str(e)}", "", "", state_data


def answer_question_gradio(video_url: str, user_question: str, state_data: dict) -> str:
    if not user_question:
        return "Please provide a valid question."

    try:
        chunks = state_data.get("chunks")

        if not chunks:
            if not video_url:
                return "Please provide a video URL first."
            transcript = fetch_transcript_from_stt(video_url)
            chunks = chunk_transcript_for_embeddings(transcript)

            state_data["video_url"] = video_url
            state_data["processed_transcript"] = transcript
            state_data["chunks"] = chunks

        if "faiss_index" not in state_data:
            logger.info("Creating FAISS index...")
            with log_time("FAISS index build"):
                state_data["faiss_index"] = FAISS.from_texts(chunks, GLOBAL_EMBEDDINGS)

        faiss_index = state_data["faiss_index"]
        context = retrieve_context(user_question, faiss_index, k=4)

        return QA_CHAIN.predict(context=context, question=user_question)

    except Exception as e:
        logger.error("QA error: %s", e)
        return f"Error: {str(e)}"


# =============================================================================
# GRADIO UI
# =============================================================================

with gr.Blocks() as interface:
    gr.Markdown("# 🎥 YouTube Video Summarizer & Q&A Tool (STT-Based)")
    gr.Markdown("This version does **not** depend on YouTube captions. It downloads audio and transcribes it locally.")

    session_state = gr.State(value={})

    video_url = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

    summarize_btn = gr.Button("🚀 Summarize Video", variant="primary")

    token_info_output = gr.Textbox(
        label="Context Info",
        lines=1,
        interactive=False
    )
    transcript_output = gr.Textbox(
        label="Transcript (STT Output)",
        lines=12,
        interactive=False
    )
    summary_output = gr.Textbox(
        label="Summary",
        lines=6,
        interactive=False
    )

    question_input = gr.Textbox(label="Ask a Question", lines=2)
    question_btn = gr.Button("✨ Get Answer", variant="primary")
    answer_output = gr.Textbox(label="Answer", lines=6)

    summarize_btn.click(
        summarize_video_gradio,
        inputs=[video_url, session_state],
        outputs=[transcript_output, summary_output, token_info_output, session_state]
    )

    question_btn.click(
        answer_question_gradio,
        inputs=[video_url, question_input, session_state],
        outputs=answer_output
    )

    gr.Markdown(
        """
        ## ⚙️ Requirements

        You must install:

        - `pip install faster-whisper`
        - `pip install yt-dlp`
        - `ffmpeg` installed and available in PATH

        """
    )

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting YouTube STT Video Summarizer & Q&A Application...")
    logger.info(f"Using model: {LLM_MODEL}")
    logger.info(f"Using embeddings: {EMBEDDING_MODEL}")

    # Ensure directories exist
    ensure_dirs()

    # Enable queueing for better concurrent request handling
    interface.queue()
    interface.launch(server_name="localhost", server_port=7860)
