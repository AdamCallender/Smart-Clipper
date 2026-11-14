"""Speech to text with word timestamps (faster-whisper). GPU first, CPU fallback."""
import os, sys
from pathlib import Path


def _add_nvidia_dlls():
    """On Windows, make pip-installed CUDA libraries (nvidia-cublas-cu12,
    nvidia-cudnn-cu12) findable by the transcription engine."""
    if os.name != "nt":
        return
    for base in map(Path, sys.path):
        nv = base / "nvidia"
        if nv.is_dir():
            for d in nv.glob("*/bin"):
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


def transcribe(wav, model_name="large-v3-turbo", device="auto", language=None, log=print):
    from faster_whisper import WhisperModel

    attempts = [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]
    if device == "cpu":
        attempts = attempts[-1:]
    elif device == "cuda":
        attempts = attempts[:2]
    _add_nvidia_dlls()

    last = None
    for dev, ctype in attempts:
        try:
            model = WhisperModel(model_name, device=dev, compute_type=ctype)
            segments, info = model.transcribe(
                str(wav), word_timestamps=True, vad_filter=True, language=language,
                vad_parameters={"min_silence_duration_ms": 400})
            out = []
            for s in segments:  # generator: the work happens here
                text = s.text.strip()
                if not text:
                    continue
                out.append({
                    "start": round(s.start, 2), "end": round(s.end, 2), "text": text,
                    "words": [{"s": round(w.start, 2), "e": round(w.end, 2),
                               "w": w.word.strip()} for w in (s.words or [])]})
                if len(out) % 50 == 0:
                    log(f"  transcribed to {s.end / 60:.1f} of {info.duration / 60:.1f} min")
            log(f"  transcription ran on {dev} ({ctype})")
            return out
        except Exception as e:  # missing CUDA libs, out of memory, etc.
            last = e
            log(f"  {dev}/{ctype} failed ({type(e).__name__}: {str(e)[:120]}); trying next option")
    raise RuntimeError(f"Transcription failed on every device: {last}")
