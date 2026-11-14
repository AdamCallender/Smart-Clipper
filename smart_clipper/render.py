"""Cut and export clips with ffmpeg: horizontal or vertical, optional burned-in captions."""
import re, subprocess
from pathlib import Path

_encoder = None


def video_encoder():
    """Use the NVIDIA hardware encoder when it works, else software x264."""
    global _encoder
    if _encoder is None:
        test = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
        _encoder = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"] if test.returncode == 0
                    else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"])
    return _encoder


def _ts(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def caption_events(transcript, start, end, max_words=4, max_chars=24):
    """Short caption chunks timed from word timestamps (or spread evenly without them)."""
    words = []
    for seg in transcript:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        if re.fullmatch(r"[\[(].*[\])]", seg["text"]):
            continue
        ws = seg.get("words")
        if not ws:
            toks = seg["text"].split()
            step = (seg["end"] - seg["start"]) / max(1, len(toks))
            ws = [{"s": seg["start"] + i * step, "e": seg["start"] + (i + 1) * step, "w": t}
                  for i, t in enumerate(toks)]
        words += [w for w in ws if w["e"] > start and w["s"] < end]
    events, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        text = " ".join(x["w"] for x in cur)
        if (nxt is None or len(cur) >= max_words or len(text) + len(nxt["w"]) > max_chars
                or nxt["s"] - w["e"] > 0.5 or re.search(r"[.!?]$", w["w"])):
            stop = min(cur[-1]["e"] + 0.15, nxt["s"]) if nxt else cur[-1]["e"] + 0.3
            events.append((cur[0]["s"] - start, max(stop, cur[0]["s"] + 0.2) - start, text))
            cur = []
    return events


def write_ass(path, events, vertical):
    w, h = (1080, 1920) if vertical else (1920, 1080)
    size, margin = (64, 500) if vertical else (54, 70)
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
            f"Style: D,Arial,{size},&H00FFFFFF,&H00000000,&H80000000,1,1,5,0,2,60,60,{margin}\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Text\n")
    body = "\n".join(f"Dialogue: 0,{_ts(a)},{_ts(b)},D,{t.replace('{', '(').replace('}', ')')}"
                     for a, b, t in events)
    Path(path).write_text(head + body + "\n", encoding="utf-8")


def render_clip(src, start, end, out_dir, name, vertical, style="fit", transcript=None):
    """Render one clip into out_dir/name.mp4. Captions are burned in when a transcript is given."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = end - start
    if vertical and style == "crop":
        vf = "crop='min(iw,ih*9/16)':ih,scale=1080:1920"
    elif vertical:
        # sharp centre of the picture over a blurred fill, so more of the frame stays visible
        vf = ("split[a][b];[a]crop='min(iw,ih*9/16)':ih,scale=1080:1920,boxblur=24:2[bg];"
              "[b]crop='min(iw,ih*9/8)':ih,scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
    else:
        vf = "scale=-2:'min(1080,ih)'"
    ass = None
    if transcript is not None:
        events = caption_events(transcript, start, end)
        if events:
            ass = f"{name}.ass"  # relative name + cwd avoids Windows path escaping in filters
            write_ass(out_dir / ass, events, vertical)
            vf += f",subtitles={ass}"
    af = f"afade=t=in:d=0.1,afade=t=out:st={max(0, dur - 0.2):.2f}:d=0.2"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
           "-i", str(Path(src).resolve()), "-filter_complex", vf, *video_encoder(),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-af", af,
           "-movflags", "+faststart", f"{name}.mp4"]
    subprocess.run(cmd, check=True, cwd=out_dir)
    if ass:
        (out_dir / ass).unlink()
    return out_dir / f"{name}.mp4"


def concat(parts, dest):
    """Join clips that were rendered with identical settings."""
    dest = Path(dest)
    lst = dest.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in parts),
                   encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(dest)], check=True)
    lst.unlink()
    return dest
