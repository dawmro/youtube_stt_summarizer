# 🎥 YouTube STT Summarizer & Timestamp-Aware Q&A Tool

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/downloads/)
[![Built with Ollama](https://img.shields.io/badge/Built%20with-Ollama-orange)](https://ollama.ai)
[![Powered by LangChain](https://img.shields.io/badge/Powered%20by-LangChain-green)](https://langchain.com)
[![UI: Gradio](https://img.shields.io/badge/UI-Gradio-red)](https://gradio.app)

A high-performance, locally-hosted tool for summarizing YouTube videos and answering questions about their content using Retrieval-Augmented Generation (RAG) with **Ollama** and **LangChain**. All processing runs on your machine — no cloud APIs, no data leaving your device.

## ✨ Features

- 🎬 **Automatic Video Summarization** — Get concise summaries of any YouTube video using local speech-to-text and a local LLM
- 🤔 **Timestamp-Aware Q&A** — Ask questions about video content and receive answers with **clickable timestamp links** that jump to the exact moment in the video
- ⚡ **Smart Three-Layer Cache** — Transcripts, summaries, and FAISS indexes are persisted to disk; re-running the same video is instant
- 📏 **Adaptive Summarization** — Short videos are summarized in one pass; long videos use hierarchical multi-pass chunk reduction to stay within LLM context limits
- 🔒 **Privacy-First** — 100% local processing, no cloud dependencies
- 👥 **Multi-User Safe** — Thread-safe session isolation for concurrent users
- ⚡ **Production-Ready** — Comprehensive logging, error handling, and full type hints throughout
- 🎨 **Clean UI** — Dual-tab Gradio interface with live progress bars for both STT and summarization stages

![App screenshot](https://github.com/dawmro/youtube_stt_summarizer/blob/main/img/view_summarize.png?raw=true)
![App screenshot](https://github.com/dawmro/youtube_stt_summarizer/blob/main/img/view_qa.png?raw=true)

---

## 🏗️ Architecture

### Processing Pipeline

Every video request flows through five ordered stages. Cached results short-circuit the pipeline at each stage independently — changing only the LLM model invalidates the summary cache but reuses the transcript cache.

```
YouTube URL
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. AUDIO ACQUISITION                                           │
│                                                                 │
│  yt-dlp ──► best-quality audio stream (webm/m4a/mp3)            │
│      │                                                          │
│      ▼                                                          │
│  ffmpeg ──► 16 kHz · mono · PCM16 WAV                           │
│                                        [cache: audio_cache/]    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. SPEECH-TO-TEXT  (faster-whisper)                            │
│                                                                 │
│  WhisperModel(small, cpu, int8)                                 │
│      │                                                          │
│      ├─► TranscriptSegment list  (text + start/end timestamps)  │
│      └─► full transcript string                                 │
│                                      [cache: transcript_cache/] │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐   ┌──────────────────────────────────────┐
│  3. SUMMARIZATION   │   │  4. RETRIEVAL INDEX BUILD            │
│                     │   │                                      │
│  Token count ≤ 7168 │   │  TranscriptSegment list              │
│    └─► direct pass  │   │      │                               │
│                     │   │      ▼                               │
│  Token count > 7168 │   │  RetrievalChunk windows              │
│    └─► chunk → merge│   │  (1000-char sliding windows,         │
│        up to 4 passes│  │   1-segment overlap)                 │
│                     │   │      │                               │
│  [cache: summary_   │   │      ▼                               │
│   cache/]           │   │  OllamaEmbeddings (mxbai-embed-large)│
│                     │   │      │                               │
│                     │   │      ▼                               │
│                     │   │  FAISS index                         │
│                     │   │  [cache: retrieval_cache/]           │
└─────────────────────┘   └──────────────────────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Q&A  (RAG)                                                  │
│                                                                 │
│  Question ──► FAISS similarity search (top-4 chunks)            │
│                   │                                             │
│                   ▼                                             │
│  Labelled context (S1…S4) + timestamps ──► Ollama LLM           │
│                   │                                             │
│                   ▼                                             │
│  Raw answer ──► citation renderer                               │
│                   │                                             │
│                   ▼                                             │
│  Markdown answer with clickable [HH:MM:SS] timestamp links      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web UI** | Gradio | Dual-tab interface with live streaming progress |
| **Audio Download** | yt-dlp | Fetches best-quality audio stream from YouTube |
| **Audio Normalization** | ffmpeg | Converts any format to 16 kHz mono PCM WAV |
| **Speech-to-Text** | faster-whisper (CTranslate2) | Local Whisper inference with segment timestamps |
| **LLM** | Ollama + Llama 3.1 8B | Summarization and question answering |
| **Embeddings** | Ollama + mxbai-embed-large | Dense vector representations for RAG |
| **Vector Store** | FAISS | Approximate nearest-neighbour similarity search |
| **Text Processing** | LangChain | Prompt templates, chunking, chain wiring |
| **Session State** | Gradio `gr.State` | Per-user transcript, chunks, and FAISS index isolation |

### Cache System

The cache uses **content-addressed filenames** — the config hash is baked into the filename, so changing any setting (model, chunk size, prompt version) automatically routes to a fresh file without manual cache busting.

```
cache/
├── audio_cache/
│   └── {video_id}/
│       └── audio.wav            # normalized 16 kHz mono WAV
├── ytdlp_cache/
│   └── {video_id}/
│       └── source.webm          # raw yt-dlp download
│
├── transcript_cache/
│   └── {video_id}{stt_config_hash}.json   # text + typed segments
│
├── summary_cache/
│   └── {video_id}{mode}{summary_config_hash}{transcript_hash}.json
│
└── retrieval_cache/
    └── {video_id}{retrieval_config_hash}{transcript_hash}/
        ├── chunks.json          # RetrievalChunk metadata
        └── index.faiss          # FAISS binary index
```

**Cache invalidation rules:**

| Change | Transcript | Summary | FAISS index |
|--------|-----------|---------|-------------|
| Different video URL | ♻️ Rebuilt | ♻️ Rebuilt | ♻️ Rebuilt |
| Change Whisper model/settings | ♻️ Rebuilt | ♻️ Rebuilt | ♻️ Rebuilt |
| Change LLM model or prompt | ✅ Reused | ♻️ Rebuilt | ✅ Reused |
| Change embedding model | ✅ Reused | ✅ Reused | ♻️ Rebuilt |
| Same video, same settings | ✅ Reused | ✅ Reused | ✅ Reused |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- [Ollama](https://ollama.ai) installed and running
- [ffmpeg](https://ffmpeg.org/download.html) installed and available in `PATH`
- ~15 GB disk space for models
- GPU recommended but optional (CUDA support)

### Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/dawmro/youtube_stt_summarizer.git
   ```

2. **Navigate into the directory**
   ```sh
   cd youtube_stt_summarizer
   ```

3. **Create a virtual environment**
   ```sh
   py -3.12 -m venv env
   ```

4. **Activate the virtual environment**
   ```sh
   # Windows
   env\Scripts\activate

   # macOS / Linux
   source env/bin/activate
   ```

5. **Install Python dependencies**
   ```sh
   pip install -r requirements.txt
   ```

6. **Pull required Ollama models**
   ```sh
   # LLM for summarization and Q&A (~9 GB)
   ollama pull llama3.1:8b-instruct-q8_0

   # Embedding model for FAISS retrieval (~670 MB)
   ollama pull mxbai-embed-large
   ```

7. **Start the Ollama server** (separate terminal)
   ```sh
   ollama serve
   ```

8. **Run the app**
   ```sh
   python app.py
   ```

9. **Open the UI in your browser**
   ```
   http://localhost:7860/
   ```

---

## 🛠️ Tests

```sh
pytest -v
```

The test suite stubs out Whisper and Ollama so no models need to be running. All unit tests cover URL parsing, time formatting, hashing, data model serialization, chunking logic, citation rendering, and session state transitions.

---

## 📖 Usage

### Basic Workflow

1. **Enter a YouTube URL**
   Paste any supported YouTube URL into the input field on the **Summarize** tab.

2. **Summarize the video**
   Click **Summarize**. The app will:
   - Download the audio with yt-dlp
   - Convert it to a 16 kHz mono WAV with ffmpeg
   - Transcribe it locally with faster-whisper
   - Summarize it with Ollama (one pass for short videos, hierarchical chunked reduction for long ones)

   Both progress bars update in real time. On repeat runs of the same video with the same settings, all stages are served from cache.

3. **Ask questions**
   Switch to the **Q&A** tab. Type your question and click **Ask**. The app:
   - Embeds your question and searches the FAISS index for the top-4 most relevant transcript chunks
   - Sends the retrieved context to the LLM with source labels (S1–S4)
   - Renders the answer with **clickable timestamp links** that open the video at the exact moment

### Supported URLs

```
✅ https://www.youtube.com/watch?v=dQw4w9WgXcQ
✅ https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s
✅ https://youtu.be/dQw4w9WgXcQ
✅ https://www.youtube.com/shorts/dQw4w9WgXcQ
❌ Private or members-only videos
❌ Live streams (no downloadable audio)
```

---

## ⚙️ Configuration

All settings are controlled by the `AppConfig` dataclass at the top of `app.py`. No environment variables or config files are required — edit the defaults directly.

### Model Selection

```python
# LLM for summarization and Q&A
llm_model: str = "llama3.1:8b-instruct-q8_0"

# Embedding model for FAISS retrieval
embedding_model: str = "mxbai-embed-large"   # or "nomic-embed-text"
```

#### Available LLM Models

```sh
# Recommended — balanced speed and quality
ollama pull llama3.1:8b-instruct-q8_0

# Higher quality, slower inference
ollama pull llama3.1:70b-instruct-q8_0

# Lighter quantization — lower RAM usage, slightly reduced quality
ollama pull llama3.1:8b-instruct-q4_0
```

#### Available Embedding Models

```sh
ollama pull mxbai-embed-large     # Recommended (335 M parameters, ~670 MB)
ollama pull nomic-embed-text      # Lightweight alternative (137 M parameters)
```

### Whisper Settings

```python
whisper_model_size: str = "small"     # tiny | base | small | medium | large-v3
whisper_device: str = "cpu"           # cpu | cuda
whisper_compute_type: str = "int8"    # int8 (CPU) | float16 (GPU)
whisper_language: Optional[str] = None  # None = auto-detect, or e.g. "en", "pl"
whisper_beam_size: int = 1            # 1 = fastest; 5 = more accurate
whisper_vad_filter: bool = False      # True strips silence before transcription
```

### Full Requirements

See `requirements.txt` for the complete dependency list.

---

## 🔄 Performance Optimization

### GPU Acceleration (Recommended)

The single biggest speedup is moving Whisper inference to a CUDA GPU. Edit `AppConfig`:

```python
whisper_device: str = "cuda"
whisper_compute_type: str = "float16"   # or "int8_float16" for lower VRAM usage
```

### Batched Inference Pipeline (CPU or GPU)

For CPU-only setups, `BatchedInferencePipeline` parallelizes the mel-spectrogram computation and inference across audio chunks, giving roughly a **2–3× speedup** with no model change:

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel(CFG.whisper_model_size, device="cpu", compute_type="int8")
whisper = BatchedInferencePipeline(model=model)

# then in transcribe_audio():
segments, info = whisper.transcribe(str(audio_path), batch_size=8, **kwargs)
```

### Whisper Model Size Trade-offs

| Model | VRAM | Relative speed | WER (English) |
|-------|------|---------------|---------------|
| `tiny` | ~1 GB | ~10× | Higher |
| `base` | ~1 GB | ~7× | Medium |
| `small` | ~2 GB | ~4× | Good *(default)* |
| `medium` | ~5 GB | ~2× | Better |
| `large-v3` | ~10 GB | 1× | Best |

### Transcription Accuracy Knobs

```python
# Enables Voice Activity Detection — strips silence before Whisper sees the audio.
# Speeds up transcription when the video has significant non-speech sections.
whisper_vad_filter: bool = True

# Increases beam search width — slower but more accurate for ambiguous speech.
whisper_beam_size: int = 5

# If you know the video language, setting it skips the auto-detection step.
whisper_language: str = "en"
```

### Cache Warm Hits

On second and subsequent runs of the same video with unchanged settings, the entire pipeline is served from disk cache. The UI reports `Using cached transcript.` and skips download, conversion, and all inference.

---

## 🐛 Troubleshooting

### Ollama Connection Failed

```sh
# Verify Ollama is reachable
curl http://localhost:11434/api/tags

# Start the server if it is not running
ollama serve
```

### Required Model Not Found

```
RuntimeError: Ollama is online, but required models are missing: [llama3.1:8b-instruct-q8_0]
```

```sh
ollama pull llama3.1:8b-instruct-q8_0
ollama pull mxbai-embed-large
```

### Out of Memory (OOM)

```sh
# Switch to a lighter quantization
ollama pull llama3.1:8b-instruct-q4_0
```

Then update `AppConfig`:

```python
llm_model: str = "llama3.1:8b-instruct-q4_0"
```

### Whisper Is Very Slow on First Run

CTranslate2 loads model weights lazily on the first transcription call. On some systems the cold-start can take 1–5 minutes. Subsequent runs are fast because the weights stay in RAM. To front-load this cost to startup, add a warmup call in `build_runtime()`:

```python
import numpy as np

def warmup_whisper(whisper: WhisperModel) -> None:
    dummy = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    segs, _ = whisper.transcribe(dummy, language="en")
    list(segs)

# inside build_runtime(), after WhisperModel(...):
warmup_whisper(whisper)
```

### ffmpeg Not Found

```
FileNotFoundError: [WinError 2] The system cannot find the file specified: 'ffmpeg'
```

Download ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it is on your system `PATH`. Verify with:

```sh
ffmpeg -version
```

### yt-dlp Download Fails

```sh
# Update yt-dlp — YouTube changes its API frequently
pip install -U yt-dlp
```

### Stale Cache After Changing Settings

The cache key is derived from the settings hash. If you change `AppConfig` values (model name, chunk size, prompt version), new cache files are created automatically and old ones are left on disk. To reclaim space:

```sh
# Windows
rmdir /s /q cache

# macOS / Linux
rm -rf cache/
```

---

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [LangChain Documentation](https://python.langchain.com/)
- [Gradio Documentation](https://gradio.app/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

---

## ❓ FAQ

**Q: Can I use cloud-hosted models instead of local Ollama?**
A: Yes. Swap the `Ollama` and `OllamaEmbeddings` instances in `build_runtime()` for any LangChain-compatible provider (OpenAI, Anthropic, etc.).

**Q: Does this work with private videos?**
A: No — only publicly accessible videos can be downloaded by yt-dlp.

**Q: Can I export summaries and Q&A results?**
A: Not built-in, but summaries are cached as plain JSON in `cache/summary_cache/` and can be read directly.

**Q: What is the maximum supported video length?**
A: There is no hard limit. The summarization pipeline uses hierarchical chunk reduction (up to 4 passes) to handle transcripts of any length within the LLM context window. Transcription time scales linearly with video duration.

**Q: Why does the first transcription take much longer than subsequent ones?**
A: CTranslate2 loads model weights from disk on the first inference call. Once loaded, weights stay resident in RAM. See the *Whisper Is Very Slow on First Run* section above for a warmup fix.

**Q: Can I run multiple videos at the same time?**
A: The Gradio queue is configured with `default_concurrency_limit=1`, meaning requests are serialized. The session state ensures each user's transcript and index are fully isolated.

---

## 🔐 Security & Privacy

- ✅ **100% Local Processing** — No audio or transcript data is sent to external servers
- ✅ **No API Keys Required** — Fully self-hosted and self-contained
- ✅ **Privacy Preserved** — All inference runs on your own hardware
- ⚠️ **Ollama Server** — By default Ollama listens on `localhost:11434`. Ensure your firewall prevents external access if running on a shared machine
