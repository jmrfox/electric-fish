"""Load the tracking + EOD CSV exports into a common, tidy structure.

Each CSV row is one electric organ discharge (EOD). A row carries:

* the video frame the EOD falls in (``Frame``) and a time,
* the pose of every fish in that frame (``{keypoint}_{x|y|score}_{fish}``),
* a 25-sample voltage snippet around the EOD peak on each of 10 electrodes,
* (newer exports only) per-EOD analysis columns such as ``discharging_fish``.

Two export formats exist and are normalised here:

``legacy`` (e.g. ``Data_4fish_2024-02-20T12_56_22.csv``)
    Fish are numbered ``1..N``, keypoints include ``chin``, ``Time`` is the EOD time
    in seconds (sub-frame resolution) and electrodes are ``electrode_{e}_time_{1..25}``.

``assigned`` (e.g. ``Assigned_Data_3fish_2024-10-15T14_12_25.csv``)
    Fish are lettered ``A..``, ``Time`` is a wall-clock timestamp, ``RelativeTime`` the
    frame time in seconds, electrodes are ``electrode_{e}_t_{m12..0..p12}``, and extra
    columns (``discharging_fish``, ``similarity_*``, lengths, pairwise distances) exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

N_ELECTRODES = 10
N_SAMPLES = 25

KEYPOINT_RE = re.compile(r"^(chin|mouth|head|middle|tail)_(x|y|score)_(\w+)$")
LEGACY_ELECTRODE_RE = re.compile(r"^electrode_(\d+)_time_(\d+)$")
ASSIGNED_ELECTRODE_RE = re.compile(r"^electrode_(\d+)_t_(m\d+|p\d+|0)$")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}")


def _sample_offset(token: str) -> int:
    """``m12`` -> -12, ``0`` -> 0, ``p05`` -> 5."""
    if token == "0":
        return 0
    sign = -1 if token[0] == "m" else 1
    return sign * int(token[1:])


def _electrode_layout(columns: list[str]) -> tuple[str, list[str]]:
    """Return the format name and electrode columns ordered as (electrode, sample)."""
    legacy = [(int(m[1]), int(m[2]), c) for c in columns if (m := LEGACY_ELECTRODE_RE.match(c))]
    assigned = [
        (int(m[1]), _sample_offset(m[2]), c)
        for c in columns
        if (m := ASSIGNED_ELECTRODE_RE.match(c))
    ]
    if legacy and not assigned:
        fmt, cols = "legacy", legacy
    elif assigned and not legacy:
        fmt, cols = "assigned", assigned
    else:
        raise ValueError("Could not determine electrode column layout")
    if len(cols) != N_ELECTRODES * N_SAMPLES:
        raise ValueError(f"Expected {N_ELECTRODES * N_SAMPLES} electrode columns, got {len(cols)}")
    return fmt, [c for _, _, c in sorted(cols)]


def recording_timestamp(path: str | Path) -> str | None:
    """Extract the ``YYYY-MM-DDTHH_MM_SS`` stamp used to pair CSVs with videos."""
    m = TIMESTAMP_RE.search(Path(path).name)
    return m.group(0) if m else None


@dataclass
class Recording:
    """One session: EOD events, electrode waveforms and per-frame fish poses.

    Attributes
    ----------
    eods
        One row per EOD. Always has ``frame`` and ``time_s`` (seconds from the start of the
        recording); the assigned format additionally keeps ``timestamp`` and its analysis
        columns (e.g. ``discharging_fish``). Index is the EOD number (0..n-1).
    waveforms
        ``float32`` array, shape ``(n_eods, 10 electrodes, 25 samples)``, aligned with ``eods``.
    poses
        One row per video frame that has at least one EOD. Columns are a MultiIndex
        ``(fish, keypoint, field)`` with field in ``x``, ``y``, ``score``.
    """

    name: str
    format: str
    eods: pd.DataFrame
    waveforms: np.ndarray
    poses: pd.DataFrame
    csv_path: Path | None = None
    video_path: Path | None = None
    n_pose_conflicts: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def fish(self) -> list[str]:
        return list(self.poses.columns.get_level_values("fish").unique())

    @property
    def keypoints(self) -> list[str]:
        return list(self.poses.columns.get_level_values("keypoint").unique())

    def __repr__(self) -> str:
        return (
            f"Recording({self.name!r}, format={self.format!r}, fish={self.fish}, "
            f"eods={len(self.eods)}, frames={len(self.poses)}, "
            f"video={'yes' if self.video_path else 'no'})"
        )

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_csv(cls, csv_path: str | Path, video_path: str | Path | None = None) -> Recording:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, engine="pyarrow")
        rec = cls.from_dataframe(df, name=recording_timestamp(csv_path) or csv_path.stem)
        rec.csv_path = csv_path
        rec.video_path = Path(video_path) if video_path else None
        return rec

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "recording") -> Recording:
        fmt, electrode_cols = _electrode_layout(list(df.columns))
        waveforms = (
            df[electrode_cols].to_numpy(np.float32).reshape(len(df), N_ELECTRODES, N_SAMPLES)
        )

        kp_cols = [c for c in df.columns if KEYPOINT_RE.match(c)]
        other_cols = [c for c in df.columns if c not in set(kp_cols) | set(electrode_cols)]

        eods = df[other_cols].rename(columns={"Frame": "frame"}).reset_index(drop=True)
        if fmt == "legacy":
            eods = eods.rename(columns={"Time": "time_s"})
        else:
            eods = eods.rename(columns={"Time": "timestamp", "RelativeTime": "time_s"})
            eods["timestamp"] = pd.to_datetime(eods["timestamp"])
        eods.index.name = "eod"

        # Pose is repeated on every EOD row of a frame; keep one row per frame.
        pose_rows = df[["Frame", *kp_cols]]
        n_conflicts = int((pose_rows.groupby("Frame").nunique(dropna=False) > 1).any(axis=1).sum())
        poses = pose_rows.groupby("Frame").first()
        poses.index.name = "frame"
        tuples = []
        for c in kp_cols:
            kp, fld, fish = KEYPOINT_RE.match(c).groups()
            tuples.append((fish, kp, fld))
        poses.columns = pd.MultiIndex.from_tuples(tuples, names=["fish", "keypoint", "field"])
        poses = poses.sort_index(axis=1)

        return cls(
            name=name,
            format=fmt,
            eods=eods,
            waveforms=waveforms,
            poses=poses,
            n_pose_conflicts=n_conflicts,
        )

    # ------------------------------------------------------------------ caching

    def save(self, directory: str | Path) -> Path:
        """Write a fast-loading cache (parquet + npy) to ``directory/<name>/``."""
        out = Path(directory) / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.eods.to_parquet(out / "eods.parquet")
        flat = self.poses.copy()
        flat.columns = ["|".join(c) for c in flat.columns]
        flat.to_parquet(out / "poses.parquet")
        np.save(out / "waveforms.npy", self.waveforms)
        pd.Series(
            {
                "format": self.format,
                "csv_path": str(self.csv_path or ""),
                "video_path": str(self.video_path or ""),
                "n_pose_conflicts": self.n_pose_conflicts,
            }
        ).to_json(out / "meta.json")
        return out

    @classmethod
    def load_cache(cls, directory: str | Path) -> Recording:
        d = Path(directory)
        meta = pd.read_json(d / "meta.json", typ="series")
        poses = pd.read_parquet(d / "poses.parquet")
        poses.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("|")) for c in poses.columns], names=["fish", "keypoint", "field"]
        )
        return cls(
            name=d.name,
            format=meta["format"],
            eods=pd.read_parquet(d / "eods.parquet"),
            waveforms=np.load(d / "waveforms.npy", mmap_mode="r"),
            poses=poses,
            csv_path=Path(meta["csv_path"]) if meta["csv_path"] else None,
            video_path=Path(meta["video_path"]) if meta["video_path"] else None,
            n_pose_conflicts=int(meta["n_pose_conflicts"]),
        )

    # ------------------------------------------------------------------ helpers

    def keypoint(self, name: str) -> pd.DataFrame:
        """x/y/score of one keypoint for all fish: columns MultiIndex ``(fish, field)``."""
        return self.poses.xs(name, axis=1, level="keypoint")

    def open_video(self):
        """Return a :class:`~electric_fish.video.VideoReader` for this recording."""
        from electric_fish.video import VideoReader

        if self.video_path is None:
            raise FileNotFoundError(f"No video paired with recording {self.name}")
        return VideoReader(self.video_path)
