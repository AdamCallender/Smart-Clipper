# Smart-Clipper

Turn a long video (YouTube video, stream VOD, show, movie) into short clips.
It analyzes the video once, lets a local AI model pick the best moments for the
mode you choose, and exports each pick as horizontal and/or vertical video.
Everything runs on your PC. No API keys, no per-use cost.

## How it works

1. **Analyze** (saved per video, so it only happens once)
   - Transcript with word timing (faster-whisper, on your GPU)
   - Audio signals: sudden volume spikes (shouts, screams, laughs)
   - Scene cuts (ffmpeg)
2. **Pick**: the transcript is split into sections of about 2.5 minutes. A local
   model running in Ollama reads each section and returns moments with a score,
   a title and a reason. Code then fits each moment to your length limit, pads
   the cut so words aren't clipped, snaps to nearby scene cuts, and removes overlaps.
3. **Render**: ffmpeg exports horizontal (up to 1080p) and/or vertical (1080x1920)
   clips, with burned-in captions on vertical by default.

## Install (Windows)

1. **Python 3.10 or newer** from python.org (tick "Add to PATH").
2. **ffmpeg**: `winget install Gyan.FFmpeg`, or download from ffmpeg.org and add it to PATH.
3. **Ollama**: `winget install Ollama.Ollama`, or download from ollama.com.
4. In this folder:
   ```
   pip install -r requirements.txt
   ```

The first run downloads the transcription model and the picking model
(`qwen3:8b`, about 5 GB). Smart-Clipper starts Ollama itself if it isn't running.

## Use

```
python -m smart_clipper "C:\path\to\video.mkv"
python -m smart_clipper "https://www.youtube.com/watch?v=..." --mode comedy --length 45 --count 8
python -m smart_clipper video.mkv --mode find --query "the part where he falls off the tower"
python -m smart_clipper video.mkv --reel 180          (also one 3-minute condensed video)
```

| Option | What it does | Default |
|---|---|---|
| `--mode` | `highlights`, `comedy`, `serious`, `character`, `story`, `find` | `highlights` |
| `--query` | What to look for, with `--mode find` | |
| `--length` | Longest clip, in seconds | 60 |
| `--count` | Number of clips | 5 |
| `--format` | `h`, `v` or `both` | `both` |
| `--vertical-style` | `fit` (centre over blurred fill) or `crop` (full-screen centre crop) | `fit` |
| `--captions` | `vertical`, `all` or `none` | `vertical` |
| `--reel SECONDS` | Also join the best moments, in order, into one video of about this length | off |
| `--model` | Ollama model used for picking | `qwen3:8b` |
| `--brain signals` | Pick from audio spikes only, no AI (quick check, or if Ollama is unavailable) | `ollama` |
| `--picks-only` | Show the picks without rendering | off |
| `--reanalyze` | Redo the saved analysis | off |

Output goes to `output/<video name>/<mode>/`. The full pick list with scores and
reasons is saved as `output/<video name>/picks_<mode>.json`.

## Status: what has and hasn't been tested

Tested (Linux, CPU only, one 16-minute stream-highlights video):
- Audio signals, scene cuts, the pick fitting/snapping/overlap logic
- Rendering: horizontal, vertical, captions, the condensed reel
- The Ollama request/response handling, against a stand-in server that mimics Ollama's API

**Not yet tested** (could not be run in the development environment):
- GPU transcription with faster-whisper on Windows
- Picking quality with a real Ollama model. Earlier test picks were made by a much
  larger model; how well `qwen3:8b` picks is the main open question.
- The NVIDIA hardware encoder path and YouTube link download

## Known limits

- Picking is driven by speech and volume. Moments with no talking and no loud sound
  are missed. A visual pass (frame descriptions) is planned.
- Vertical framing is a fixed centre layout. Facecam detection and speaker tracking are planned.
- "Whole movie in a short" (`--mode story --reel 60`) joins key moments but has no narration.

## Troubleshooting

- **Transcription falls back to CPU**: the log says which device ran. If it reports a CUDA
  error, check that `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` installed, and update your
  NVIDIA driver.
- **Picks are weak**: try another model, e.g. `--model gemma3:12b` (about 8 GB; tight on a
  10 GB card). Run with `--picks-only` to compare models quickly, since analysis is saved.
- **Out of GPU memory during picking**: close games/browsers using the GPU, or use a smaller model.
