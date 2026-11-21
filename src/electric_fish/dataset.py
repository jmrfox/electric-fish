"""Discover recordings in the data directory and load them (with caching)."""

from __future__ import annotations

from pathlib import Path

from electric_fish.recording import Recording, recording_timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


def find_recordings(data_dir: str | Path = DATA_DIR) -> dict[str, dict[str, Path | None]]:
    """Pair each CSV with its video by the timestamp in the filename.

    Returns ``{timestamp: {"csv": path, "video": path or None}}`` sorted by timestamp.
    """
    data_dir = Path(data_dir)
    videos = {recording_timestamp(p): p for p in data_dir.glob("*.avi")}
    found = {}
    for csv in sorted(data_dir.glob("*.csv")):
        ts = recording_timestamp(csv)
        if ts is None:
            continue
        if ts in found:
            raise ValueError(f"Multiple CSVs for recording {ts}: {found[ts]['csv']}, {csv}")
        found[ts] = {"csv": csv, "video": videos.get(ts)}
    return dict(sorted(found.items()))


def load_recording(
    name: str,
    data_dir: str | Path = DATA_DIR,
    cache_dir: str | Path | None = CACHE_DIR,
    refresh: bool = False,
) -> Recording:
    """Load a recording by timestamp (e.g. ``"2024-02-20T12_56_22"``).

    The first load parses the CSV (slow for the larger files) and writes a cache to
    ``cache_dir``; later loads read the cache. Pass ``refresh=True`` to rebuild it,
    or ``cache_dir=None`` to bypass caching entirely.
    """
    if cache_dir is not None and not refresh:
        cached = Path(cache_dir) / name
        if (cached / "meta.json").exists():
            return Recording.load_cache(cached)

    recordings = find_recordings(data_dir)
    if name not in recordings:
        raise KeyError(f"Unknown recording {name!r}; available: {list(recordings)}")
    paths = recordings[name]
    rec = Recording.from_csv(paths["csv"], video_path=paths["video"])
    if cache_dir is not None:
        rec.save(cache_dir)
    return rec
