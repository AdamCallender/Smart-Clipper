"""Smart-Clipper command line: long video in, short clips out."""
import argparse, json, re, sys, time
from pathlib import Path

from . import media, modes, ollama, picker, render


def cached(path, make, fresh, log, label):
    if path.exists() and not fresh:
        log(f"{label}: using saved result")
        return json.loads(path.read_text(encoding="utf-8"))
    t = time.time()
    log(f"{label}: working...")
    data = make()
    path.write_text(json.dumps(data), encoding="utf-8")
    log(f"{label}: done in {time.time() - t:.0f}s")
    return data


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40] or "clip"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="smart-clipper", description="Turn a long video into short clips.")
    ap.add_argument("video", help="video file or YouTube/Twitch link")
    ap.add_argument("--mode", choices=list(modes.MODES), default="highlights")
    ap.add_argument("--query", help='what to look for, with --mode find (e.g. "the boss fight")')
    ap.add_argument("--length", type=float, default=60, help="longest clip, in seconds (default 60)")
    ap.add_argument("--min-length", type=float, help="shortest clip (default: a quarter of --length, at least 8)")
    ap.add_argument("--count", type=int, default=5, help="how many clips to make (default 5)")
    ap.add_argument("--format", choices=["h", "v", "both"], default="both",
                    help="horizontal, vertical or both (default both)")
    ap.add_argument("--vertical-style", choices=["fit", "crop"], default="fit",
                    help="fit: centre over blurred fill; crop: full-screen centre crop")
    ap.add_argument("--captions", choices=["vertical", "all", "none"], default="vertical")
    ap.add_argument("--reel", type=float, metavar="SECONDS",
                    help="also make one condensed video of about this many seconds")
    ap.add_argument("--brain", choices=["ollama", "signals"], default="ollama",
                    help="ollama: local AI picks; signals: audio spikes only, no AI")
    ap.add_argument("--model", default="qwen3:8b", help="Ollama model (default qwen3:8b)")
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--language", help="spoken language code, e.g. en (default: detect)")
    ap.add_argument("--out", default="output", help="output folder (default ./output)")
    ap.add_argument("--no-scenes", action="store_true", help="skip scene-cut detection")
    ap.add_argument("--reanalyze", action="store_true", help="ignore saved analysis")
    ap.add_argument("--picks-only", action="store_true", help="list picks, render nothing")
    a = ap.parse_args(argv)
    log = lambda *x, **k: print(*x, **k, flush=True)

    if a.mode == "find" and not a.query:
        ap.error("--mode find needs --query")
    media.require_ffmpeg()
    out_root = Path(a.out)

    src = Path(a.video)
    if media.is_url(a.video):
        log("Downloading video...")
        src = media.download(a.video, out_root / "downloads")
    if not src.exists():
        sys.exit(f"File not found: {src}")

    work = out_root / slug(src.stem)
    cache = work / "analysis"
    cache.mkdir(parents=True, exist_ok=True)
    info = media.probe(src)
    log(f"Video: {src.name} ({info['duration'] / 60:.1f} min, {info['width']}x{info['height']})")

    # 1. Analysis (saved, so later runs with other modes skip straight to picking)
    wav = cache / "audio.wav"
    if not wav.exists() or a.reanalyze:
        media.extract_audio(src, wav)
    signals = cached(cache / "signals.json", lambda: media.audio_signals(wav),
                     a.reanalyze, log, "Audio signals")
    transcript = cached(
        cache / "transcript.json",
        lambda: __import__("smart_clipper.transcribe", fromlist=["x"]).transcribe(
            wav, a.whisper_model, a.device, a.language, log),
        a.reanalyze, log, "Transcript")
    cuts = [] if a.no_scenes else cached(
        cache / "cuts.json", lambda: media.scene_cuts(src), a.reanalyze, log, "Scene cuts")

    lines = picker.tag_lines(picker.to_lines(transcript), signals)
    if not lines:
        sys.exit("No speech found in this video. Picking currently needs spoken audio.")

    # 2. Picking
    max_len = a.length
    min_len = a.min_length or max(8.0, a.length / 4)
    want = max(a.count, 12) if a.reel else a.count
    brain = a.brain
    if brain == "ollama":
        if not ollama.ensure_running(log):
            log("Ollama is not installed or would not start. Falling back to audio signals only.")
            brain = "signals"
        else:
            ollama.ensure_model(a.model, log)
    log(f"Picking ({a.mode}, {'model ' + a.model if brain == 'ollama' else 'audio signals only'})...")
    if brain == "ollama":
        found = picker.pick_with_ollama(lines, a.mode, a.query, a.model, min_len, max_len, log=log)
    else:
        found = picker.pick_with_signals(lines, signals, max_len, want)
    picks = picker.finalize(found, lines, signals, cuts, info["duration"], min_len, max_len, want)
    if not picks:
        sys.exit("Nothing was picked. Try another mode or a longer --length.")

    clips = picks[:a.count]
    for n, p in enumerate(clips, 1):
        p["name"] = f"{n:02d}_{slug(p['title'])}"
    tag = a.mode + ("" if brain == "ollama" else "_signals")
    (work / f"picks_{tag}.json").write_text(json.dumps(picks, indent=1), encoding="utf-8")

    log("")
    for n, p in enumerate(clips, 1):
        log(f"{n}. [{p['rank_score']:.1f}] {p['start'] // 60:.0f}:{p['start'] % 60:04.1f}"
            f" -> {p['end'] // 60:.0f}:{p['end'] % 60:04.1f} ({p['end'] - p['start']:.0f}s)  {p['title']}")
        log(f"     {p['reason']}")
    if a.picks_only:
        return

    # 3. Rendering
    log("")
    fmts = ["h", "v"] if a.format == "both" else [a.format]
    cap = lambda vertical: transcript if (a.captions == "all" or (a.captions == "vertical" and vertical)) else None
    for p in clips:
        for f in fmts:
            v = f == "v"
            dst = render.render_clip(src, p["start"], p["end"], work / tag, f"{p['name']}_{f}",
                                     v, a.vertical_style, cap(v))
            log(f"Saved {dst}")

    if a.reel:
        reel = picker.build_reel(picks, a.reel)
        for f in fmts:
            v = f == "v"
            parts = [render.render_clip(src, p["start"], p["end"], work / tag / "_parts",
                                        f"part{i:02d}_{f}", v, a.vertical_style, cap(v))
                     for i, p in enumerate(reel)]
            dst = render.concat(parts, work / tag / f"reel_{int(a.reel)}s_{f}.mp4")
            for part in parts:
                part.unlink()
            log(f"Saved {dst} ({len(reel)} moments, {sum(p['end'] - p['start'] for p in reel):.0f}s)")
        try:
            (work / tag / "_parts").rmdir()
        except OSError:
            pass
    log("\nDone.")


if __name__ == "__main__":
    main()
