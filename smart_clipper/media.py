"""ffmpeg helpers: download, probe, audio extraction, scene cuts, audio signals."""
import json, re, shutil, subprocess, sys, wave
from pathlib import Path

import numpy as np


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def require_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH. See the README install steps.")


def is_url(s):
    return bool(re.match(r"https?://", s))


def download(url, dest_dir):
    """Download a video with yt-dlp, capped at 1080p. Returns the file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
         "--merge-output-format", "mkv", "--no-playlist", "--restrict-filenames",
         "-o", str(dest_dir / "%(title).80s_%(id)s.%(ext)s"),
         "--print", "after_move:filepath", url],
        check=True, capture_output=True, text=True)
    return Path(out.stdout.strip().splitlines()[-1])


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    return {"width": int(s["width"]), "height": int(s["height"]),
            "duration": float(j["format"]["duration"])}


def extract_audio(src, wav):
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", wav])


def load_wav(wav):
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr


def audio_signals(wav):
    """Per-second loudness (dB) and 'jump': rise over the previous 5 s.
    Jumps mark outbursts, screams, laughter and stingers."""
    a, sr = load_wav(wav)
    n = len(a) // sr
    if n == 0:
        return {"db": [], "jump": []}
    frames = a[: n * sr].reshape(n, sr)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1) + 1e-12) + 1e-9)
    prev = np.array([db[max(0, i - 5):i].mean() if i else db[0] for i in range(n)])
    return {"db": db.round(1).tolist(), "jump": (db - prev).round(1).tolist()}


def scene_cuts(src, threshold=0.3):
    """Timestamps of hard cuts, found by ffmpeg on a small low-fps copy of the picture."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-hwaccel", "auto", "-i", str(src), "-an",
         "-vf", f"fps=4,scale=320:-2,select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True)
    return [round(float(m), 2) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)]
