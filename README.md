# electric-fish

Modeling the behavior and electric organ discharges (EODs) of weakly electric fish (Mormyridae)
from synchronized video and electrode recordings.

## Setup

```bash
uv sync
uv run python scripts/build_cache.py   # parse CSVs once into data/cache/ (parquet + npy)
uv run pytest
```

Notebooks in `notebooks/` are [marimo](https://marimo.io) notebooks: reactive, stored as plain
`.py` files, with sliders and dropdowns for interactive exploration.

```bash
uv run marimo edit notebooks/voltage_viewer.py   # edit in the browser
uv run marimo edit notebooks/tracking_qc.py      # tracking quality: swaps, jumps, video overlay
uv run marimo run notebooks/voltage_viewer.py    # use as a read-only app
```

For scrubbing video side by side with poses and EOD voltage, there is a GPU-rendered
[fastplotlib](https://www.fastplotlib.org) viewer in its own optional dependency group:

```bash
uv run --group viz python scripts/video_viewer.py 2024-02-13T14_33_21
```

It opens a native window (WSLg on WSL2). On WSL2 wgpu currently only finds the CPU renderer
(llvmpipe), which still manages ~25 frames/s. fastplotlib's notebook support is JupyterLab
only, so it is not embedded in the marimo notebooks.

For quick non-interactive exploration, plain `.py` files with `# %%` cells also run in
[Zed's REPL](https://zed.dev/docs/repl) (`repl: run`), using the project `.venv` as the kernel.

Raw data lives in `data/` (git-ignored): one `.avi` video and one `.csv` per session, paired by
the `YYYY-MM-DDTHH_MM_SS` timestamp in the filenames.

## Usage

```python
from electric_fish import find_recordings, load_recording

find_recordings()                       # {timestamp: {"csv": ..., "video": ...}}
rec = load_recording("2024-02-20T12_56_22")

rec.eods          # one row per EOD: frame, time_s (+ timestamp, discharging_fish, ... in newer exports)
rec.waveforms     # (n_eods, 10 electrodes, 25 samples) voltage snippets, aligned with rec.eods
rec.poses         # one row per frame; columns (fish, keypoint, x|y|score)
rec.keypoint("head")

with rec.open_video() as video:
    img = video[1234]                   # RGB frame, 912x912, 50 fps
```

## Data formats

| | legacy (Feb 2024) | assigned (Oct 2024) |
|---|---|---|
| fish ids | `1..4` | `A..C` |
| keypoints | chin, mouth, head, middle, tail | mouth, head, middle, tail |
| `time_s` | EOD time (sub-frame) | frame time (`RelativeTime`) |
| extra | — | `discharging_fish`, `similarity_*`, fish lengths, pairwise distances |

CSV `Frame` k is video index k (verified with `tracking.frame_alignment`). In the
assigned export 84 frames carry two different poses across their EOD rows (`rec.poses` keeps
the first, `rec.n_pose_conflicts` counts them).

## Layout

```
src/electric_fish/   recording.py (CSV → Recording), video.py (VideoReader), dataset.py (discovery, cache)
                     eod.py (waveform measures), tracking.py (pose QC), plotting.py
scripts/             one-off / batch scripts
notebooks/           marimo notebooks (interactive exploration)
tests/
```
