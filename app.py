import logging
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"


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
            except FileNotFoundError:
                pass


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


def download_audio(video_url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_matching_files(output_dir, "source.")
    out_template = str(output_dir / "source.%(ext)s")
    run_command(["yt-dlp", "-f", "bestaudio/best", "-o", out_template, video_url])
    candidates = sorted(output_dir.glob("source.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("yt-dlp did not produce an audio file.")
    return candidates[-1]


def convert_to_wav_16k_mono(input_audio: Path, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-i", str(input_audio),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output_wav),
    ])
    return output_wav

video_url = "https://www.youtube.com/watch?v=BSuAgw8Lc1Y"
source_dir = Path("./cache/yt_dlp_cache")
wav_path = Path("./cache/audio_cache/audio.wav")

video_id = get_video_id(video_url)
source_audio = download_audio(video_id, source_dir)
convert_to_wav_16k_mono(source_audio, wav_path)

