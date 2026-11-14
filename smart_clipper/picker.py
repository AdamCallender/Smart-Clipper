"""Choose clips: audio signals narrow the field, the editor model judges, code fits the cut."""
import re

from . import modes, ollama

LOUD_DB = 10.0  # a 1-second rise of this many dB over the previous 5 s counts as a spike


def to_lines(transcript):
    """Split transcript segments into sentence-sized lines with start/end times."""
    lines = []
    for seg in transcript:
        words = seg.get("words") or []
        if not words:
            lines.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            continue
        cur = []
        for i, w in enumerate(words):
            cur.append(w)
            gap = words[i + 1]["s"] - w["e"] if i + 1 < len(words) else 0
            if (re.search(r"[.!?]$", w["w"]) or gap > 0.7
                    or w["e"] - cur[0]["s"] > 12 or i + 1 == len(words)):
                lines.append({"start": cur[0]["s"], "end": cur[-1]["e"],
                              "text": " ".join(x["w"] for x in cur)})
                cur = []
    return [l for l in lines if l["text"] and not re.fullmatch(r"[\[(].*[\])]", l["text"])]


def peak_jump(signals, start, end):
    j = signals["jump"][max(0, int(start) - 1): int(end) + 2]
    return max(j) if j else 0.0


def tag_lines(lines, signals):
    for i, l in enumerate(lines):
        l["i"] = i
        l["jump"] = peak_jump(signals, l["start"], l["end"])
        l["loud"] = l["jump"] >= LOUD_DB
    return lines


def chunks(lines, span=150, overlap=30):
    """Overlapping windows of lines, about `span` seconds each."""
    out, i = [], 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1]["end"] - lines[i]["start"] <= span:
            j += 1
        out.append(lines[i:j + 1])
        if j + 1 >= len(lines):
            break
        k = j
        while k > i + 1 and lines[j]["end"] - lines[k]["start"] < overlap:
            k -= 1
        i = max(k, i + 1)
    return out


def fit_length(lines, a, b, min_len, max_len):
    """Adjust a line range to the length limits. Too long: drop lines from the start
    (the payoff is usually at the end). Too short: add earlier lines for setup."""
    dur = lambda: lines[b]["end"] - lines[a]["start"]
    while dur() > max_len and a < b:
        a += 1
    while dur() < min_len and a > 0 and lines[b]["end"] - lines[a - 1]["start"] <= max_len \
            and lines[a]["start"] - lines[a - 1]["end"] < 6:
        a -= 1
    return a, b


def to_times(lines, a, b, cuts, total):
    """Line range -> cut times, padded so words are not clipped, snapped to scene cuts."""
    start = max(0.0, lines[a]["start"] - 0.25)
    nxt = lines[b + 1]["start"] if b + 1 < len(lines) else total
    end = min(total, lines[b]["end"] + 0.4, max(lines[b]["end"] + 0.1, nxt - 0.05))
    prev_end = lines[a - 1]["end"] if a > 0 else 0.0
    for c in cuts:
        if prev_end <= c <= lines[a]["start"] and start - c <= 1.5:
            start = c  # start on the hard cut just before the first word
        if lines[b]["end"] <= c <= nxt and c - end <= 1.5:
            end = c    # end on the hard cut just after the last word
    return round(start, 2), round(end, 2)


def pick_with_ollama(lines, mode, query, model, min_len, max_len, per_chunk=3, log=print):
    brief = modes.MODES[mode].format(query=query or "")
    found = []
    cs = chunks(lines)
    for n, ch in enumerate(cs, 1):
        text = "\n".join(
            f"{l['i']} [{l['start']:.0f}s]{' [LOUD]' if l['loud'] else ''} {l['text']}" for l in ch)
        prompt = modes.PROMPT.format(brief=brief, per_chunk=per_chunk, min_len=int(min_len),
                                     max_len=int(max_len), lines=text)
        try:
            reply = ollama.chat_json(model, modes.SYSTEM, prompt, modes.SCHEMA)
        except Exception as e:
            log(f"  section {n}/{len(cs)}: model error ({e}); skipped")
            continue
        lo, hi = ch[0]["i"], ch[-1]["i"]
        kept = 0
        for m in reply.get("moments", [])[:per_chunk]:
            try:
                a, b, score = int(m["start_line"]), int(m["end_line"]), int(m["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (lo <= a <= b <= hi):
                continue  # the model pointed outside the section it was shown
            found.append({"a": a, "b": b, "score": max(1, min(10, score)),
                          "title": str(m.get("title", ""))[:80],
                          "reason": str(m.get("reason", ""))[:300]})
            kept += 1
        log(f"  section {n}/{len(cs)}: {kept} moment(s)")
    return found


def pick_with_signals(lines, signals, max_len, count):
    """No-model fallback: build clips around the biggest volume spikes."""
    jumps = sorted(((j, t) for t, j in enumerate(signals["jump"])), reverse=True)
    peaks = []
    for j, t in jumps:
        if j < 6 or len(peaks) >= count * 3:
            break
        if all(abs(t - p) > max_len for _, p in peaks):
            peaks.append((j, t))
    found = []
    for j, t in peaks:
        near = min(lines, key=lambda l: 0 if l["start"] <= t <= l["end"]
                   else min(abs(l["start"] - t), abs(l["end"] - t)))
        a = b = near["i"]
        while a > 0 and t - lines[a - 1]["start"] <= max_len * 0.6:
            a -= 1
        while b + 1 < len(lines) and lines[b + 1]["end"] - t <= max_len * 0.4:
            b += 1
        found.append({"a": a, "b": b, "score": round(min(9.0, 3 + j / 4), 1),
                      "title": near["text"][:60],
                      "reason": f"Volume spike of {j:.0f} dB at {t}s."})
    return found


def finalize(found, lines, signals, cuts, total, min_len, max_len, count):
    """Fit lengths, add the audio bonus, drop overlaps, keep the top `count`."""
    out = []
    for f in found:
        a, b = fit_length(lines, f["a"], f["b"], min_len, max_len)
        start, end = to_times(lines, a, b, cuts, total)
        if end - start < 3:
            continue
        bonus = min(1.0, max(0.0, peak_jump(signals, start, end)) / 15)
        out.append({**f, "a": a, "b": b, "start": start, "end": end,
                    "rank_score": round(f["score"] + bonus, 2),
                    "text": " ".join(l["text"] for l in lines[a:b + 1])})
    out.sort(key=lambda p: -p["rank_score"])
    kept = []
    for p in out:
        clash = False
        for k in kept:
            inter = min(p["end"], k["end"]) - max(p["start"], k["start"])
            if inter > 0.3 * min(p["end"] - p["start"], k["end"] - k["start"]):
                clash = True
        if not clash:
            kept.append(p)
        if len(kept) == count:
            break
    return kept


def build_reel(picks, seconds):
    """Best picks that fit in `seconds`, in the order they happen."""
    chosen, used = [], 0.0
    for p in sorted(picks, key=lambda p: -p["rank_score"]):
        d = p["end"] - p["start"]
        if used + d <= seconds:
            chosen.append(p)
            used += d
    return sorted(chosen, key=lambda p: p["start"])
