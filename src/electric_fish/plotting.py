"""Matplotlib views of EOD voltage data.

Colors follow a fixed palette: categorical slots in fixed order (one per fish, never cycled),
one blue sequential ramp for magnitude, and a blue <-> gray <-> red diverging map for signed
voltage. Ten electrodes are too many for distinguishable hues, so per-electrode views use
small multiples or a heatmap instead of ten colored lines.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from electric_fish.eod import SAMPLE_OFFSETS, peak_to_peak, strongest_trace
from electric_fish.recording import Recording

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

#: Categorical slots, assigned in this order and never cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fmt: skip

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "ef_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "ef_diverging", ["#104281", "#3987e5", NEUTRAL, "#e34948", "#8c1c1c"]
)


def use_style() -> None:
    """Apply the project's matplotlib style (thin marks, recessive grid and axes)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def fish_colors(fish: list[str]) -> dict[str, str]:
    """Stable fish -> color mapping (slot order follows the recording's fish order)."""
    if len(fish) > len(CATEGORICAL):
        raise ValueError("More fish than categorical slots")
    return dict(zip(fish, CATEGORICAL, strict=False))


def _eod_title(rec: Recording, i: int) -> str:
    row = rec.eods.iloc[i]
    title = f"EOD {i}  ·  frame {row['frame']}  ·  t = {row['time_s']:.3f} s"
    if "discharging_fish" in row:
        title += f"  ·  fish {row['discharging_fish']}"
    return title


def plot_eod(rec: Recording, i: int) -> plt.Figure:
    """One EOD on all 10 electrodes: small multiples with a shared voltage axis."""
    w = np.asarray(rec.waveforms[i])
    n = w.shape[0]
    fig, axes = plt.subplots(2, n // 2, figsize=(11, 3.8), sharex=True, sharey=True)
    p2p = peak_to_peak(w[None])[0]
    best = int(np.nanargmax(p2p)) if not np.isnan(p2p).all() else -1
    for e, ax in enumerate(axes.flat):
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(SAMPLE_OFFSETS, w[e], color=CATEGORICAL[0], lw=1.5)
        label = f"electrode {e + 1}" + ("  (strongest)" if e == best else "")
        ax.set_title(label, fontsize=9, fontweight="bold" if e == best else "normal")
    for ax in axes[-1]:
        ax.set_xlabel("sample from peak")
    for ax in axes[:, 0]:
        ax.set_ylabel("voltage")
    fig.suptitle(_eod_title(rec, i), x=0.01, ha="left", fontweight="bold")
    return fig


def plot_eod_heatmap(rec: Recording, start: int, count: int = 200, electrode: int | None = None):
    """Consecutive EODs as rows of a signed heatmap.

    ``electrode`` is 1-based; by default each row shows that EOD's strongest electrode,
    normalized so shape (not amplitude/polarity) is compared.
    """
    sl = slice(start, start + count)
    if electrode is None:
        data = strongest_trace(rec.waveforms[sl], normalize=True)
        label, what = "normalized voltage", "strongest electrode, normalized"
    else:
        data = np.asarray(rec.waveforms[sl, electrode - 1])
        label, what = "voltage", f"electrode {electrode}"
    lim = np.nanpercentile(np.abs(data), 99) or 1.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=DIVERGING,
        norm=TwoSlopeNorm(0, -lim, lim),
        extent=(SAMPLE_OFFSETS[0] - 0.5, SAMPLE_OFFSETS[-1] + 0.5, sl.start + len(data), sl.start),
        interpolation="nearest",
    )
    ax.grid(False)
    ax.set_xlabel("sample from peak")
    ax.set_ylabel("EOD index")
    ax.set_title(f"EODs {sl.start}–{sl.start + len(data) - 1} ({what})")
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    return fig


def amplitude_over_time(rec: Recording, bin_s: float = 5.0) -> pd.DataFrame:
    """Median peak-to-peak amplitude per electrode in time bins (rows: bin start, cols: 1..10)."""
    p2p = pd.DataFrame(peak_to_peak(rec.waveforms), columns=np.arange(1, 11))
    bins = (rec.eods["time_s"].to_numpy() // bin_s) * bin_s
    return p2p.groupby(bins).median()


def plot_amplitude_over_time(rec: Recording, bin_s: float = 5.0) -> plt.Figure:
    """Electrode × time heatmap of median EOD amplitude: where the signal is strongest."""
    amp = amplitude_over_time(rec, bin_s)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(
        amp.T.to_numpy(),
        aspect="auto",
        cmap=SEQUENTIAL,
        extent=(amp.index[0], amp.index[-1] + bin_s, 10.5, 0.5),
        interpolation="nearest",
        vmin=0,
    )
    ax.grid(False)
    ax.set_yticks(range(1, 11))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("electrode")
    ax.set_title(f"Median EOD peak-to-peak amplitude per electrode ({bin_s:g} s bins)")
    fig.colorbar(im, ax=ax, label="peak-to-peak voltage", pad=0.01)
    return fig


def plot_eod_rate(rec: Recording, bin_s: float = 5.0) -> plt.Figure:
    """EOD rate over time; split by discharging fish when the export assigns EODs."""
    t = rec.eods["time_s"]
    edges = np.arange(np.floor(t.min()), t.max() + bin_s, bin_s)
    centers = edges[:-1] + bin_s / 2
    fig, ax = plt.subplots(figsize=(11, 3))
    if "discharging_fish" in rec.eods:
        colors = fish_colors(rec.fish)
        for fish, color in colors.items():
            counts, _ = np.histogram(t[rec.eods["discharging_fish"] == fish], edges)
            ax.plot(centers, counts / bin_s, color=color, lw=1.5, label=f"fish {fish}")
        ax.legend(loc="upper left", ncols=len(colors))
        ax.set_title(f"EOD rate per discharging fish ({bin_s:g} s bins)")
    else:
        counts, _ = np.histogram(t, edges)
        ax.plot(centers, counts / bin_s, color=CATEGORICAL[0], lw=1.5)
        ax.set_title(f"EOD rate, all fish combined ({bin_s:g} s bins)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("EODs per second")
    ax.set_ylim(bottom=0)
    return fig


def plot_interval_histogram(rec: Recording, max_ms: float = 250.0) -> plt.Figure:
    """Histogram of intervals between consecutive EODs (per fish when assigned)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    if "discharging_fish" in rec.eods:
        groups = {f"fish {f}": g for f, g in rec.eods.groupby("discharging_fish")["time_s"]}
        colors = list(fish_colors(rec.fish).values())
    else:
        groups = {"all fish": rec.eods["time_s"]}
        colors = [CATEGORICAL[0]]
    bins = np.arange(0, max_ms + 2, 2)
    for (label, t), color in zip(groups.items(), colors, strict=False):
        iei = np.diff(np.sort(t.to_numpy())) * 1000
        ax.hist(iei, bins=bins, histtype="step", lw=1.5, color=color, label=label)
    if len(groups) > 1:
        ax.legend()
    ax.set_xlabel("interval to next EOD (ms)")
    ax.set_ylabel("count")
    ax.set_title("Inter-EOD intervals")
    return fig


def plot_waveform_overlay(rec: Recording, n: int = 400, seed: int = 0) -> plt.Figure:
    """Shape of many EODs (strongest electrode, normalized) with the median on top.

    For assigned exports there is one panel per discharging fish.
    """
    traces = strongest_trace(rec.waveforms, normalize=True)
    if "discharging_fish" in rec.eods:
        groups = {f: np.flatnonzero(rec.eods["discharging_fish"].to_numpy() == f) for f in rec.fish}
        colors = fish_colors(rec.fish)
    else:
        groups = {"all": np.arange(len(traces))}
        colors = {"all": CATEGORICAL[0]}
    rng = np.random.default_rng(seed)
    width = max(3.6 * len(groups), 5.5)
    fig, axes = plt.subplots(1, len(groups), figsize=(width, 3.2), sharey=True, squeeze=False)
    for ax, (key, idx) in zip(axes[0], groups.items(), strict=True):
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False) if len(idx) else idx
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(SAMPLE_OFFSETS, traces[pick].T, color=colors[key], lw=0.6, alpha=0.08)
        ax.plot(SAMPLE_OFFSETS, np.nanmedian(traces[idx], axis=0), color=INK, lw=2, label="median")
        who = "all EODs" if key == "all" else f"fish {key}"
        ax.set_title(f"{who}  (n = {len(idx):,})")
        ax.set_xlabel("sample from peak")
    axes[0, 0].set_ylabel("voltage / peak")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("EOD waveform shape, strongest electrode", x=0.01, ha="left", fontweight="bold")
    return fig


# --------------------------------------------------------------------------- tracking QC

EVENT_STYLE = {  # kind -> (color slot, marker); kinds are their own entities in QC charts
    "swap": (CATEGORICAL[7], "x"),
    "jump": (CATEGORICAL[6], "^"),
    "length": (CATEGORICAL[3], "o"),
}


def draw_poses(ax, rec: Recording, frame: int, labels: bool = True) -> None:
    """Overlay every fish's skeleton for one CSV frame on an existing image axis."""
    from electric_fish.tracking import BODY_ORDER

    colors = fish_colors(rec.fish)
    row = rec.poses.loc[frame]
    for fish, color in colors.items():
        kps = [k for k in BODY_ORDER if k in rec.keypoints]
        x = np.array([row[(fish, k, "x")] for k in kps], float)
        y = np.array([row[(fish, k, "y")] for k in kps], float)
        ax.plot(x, y, color=SURFACE, lw=3.5, solid_capstyle="round")  # surface ring
        ax.plot(x, y, color=color, lw=2, solid_capstyle="round")
        ax.scatter(x, y, s=18, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
        head = kps.index("head")
        if labels and np.isfinite(x[head]):
            ax.annotate(
                f"fish {fish}",
                (x[head], y[head]),
                xytext=(6, -6),
                textcoords="offset points",
                color=INK,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=color, lw=1),
            )


def plot_tracking_frames(
    rec: Recording,
    video,
    frames: list[int],
    frame_offset: int = 0,
    center: tuple[float, float] | None = None,
    size: int = 400,
) -> plt.Figure:
    """Video frames with tracked poses overlaid, side by side.

    ``frame_offset`` maps CSV frame k to video index k + offset (the alignment is unverified:
    compare 0 and -1). ``center``/``size`` crop all panels to the same square.
    """
    fig, axes = plt.subplots(1, len(frames), figsize=(3.6 * len(frames), 3.9), squeeze=False)
    h, w = video.shape
    for ax, frame in zip(axes[0], frames, strict=True):
        ax.imshow(video[frame + frame_offset], cmap="gray")
        if frame in rec.poses.index:
            draw_poses(ax, rec, frame)
        else:
            ax.text(0.5, 0.02, "no tracking for this frame", transform=ax.transAxes,
                    ha="center", color=INK, bbox=dict(fc=SURFACE, ec=AXIS))  # fmt: skip
        if center is not None:
            cx, cy = center
            ax.set_xlim(max(cx - size / 2, 0), min(cx + size / 2, w))
            ax.set_ylim(min(cy + size / 2, h), max(cy - size / 2, 0))
        ax.set_title(f"frame {frame}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    return fig


def plot_event_timeline(rec: Recording, events: pd.DataFrame) -> plt.Figure:
    """Raster of flagged tracking events over time, one row per fish."""
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rec.fish) + 1.4))
    y_of = {f: i for i, f in enumerate(rec.fish)}
    kinds = [k for k in EVENT_STYLE if k in set(events["kind"])]
    for i, kind in enumerate(kinds):
        color, marker = EVENT_STYLE[kind]
        ev = events[events["kind"] == kind]
        # offset kinds vertically within each fish row so they don't overplot
        y = ev["fish"].map(y_of) + (i - (len(kinds) - 1) / 2) * 0.22
        ax.scatter(ev["time_s"], y, marker=marker, s=22, color=color, linewidth=1.2,
                   label=f"{kind} ({len(ev):,})")  # fmt: skip
    ax.set_yticks(range(len(rec.fish)), [f"fish {f}" for f in rec.fish])
    ax.set_ylim(len(rec.fish) - 0.5, -0.5)
    ax.set_xlabel("time (s)")
    ax.set_title("Flagged tracking events")
    if kinds:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def plot_speed_histogram(rec: Recording, max_speed: float = 100.0) -> plt.Figure:
    """Per-fish distribution of frame-to-frame displacement (consecutive frames only)."""
    from electric_fish.tracking import pose_array, transition_costs

    xy, _, fish, _ = pose_array(rec)
    gap = np.diff(rec.poses.index.to_numpy())
    speed = np.diagonal(transition_costs(xy), axis1=1, axis2=2)[gap == 1]
    top = max(np.nanmax(speed), max_speed * 1.1)
    bins = np.linspace(0, top, 120)
    fig, ax = plt.subplots(figsize=(6, 3))
    for a, (f, color) in enumerate(fish_colors(fish).items()):
        ax.hist(speed[:, a], bins=bins, histtype="step", lw=1.5, color=color, label=f"fish {f}")
    ax.axvline(max_speed, color=INK_MUTED, lw=1, ls="--")
    ax.annotate(
        "jump threshold",
        (max_speed, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        color=INK_SECONDARY,
    )
    ax.set_yscale("log")
    ax.set_xlabel("mean keypoint displacement (px/frame)")
    ax.set_ylabel("frames")
    ax.set_title("Frame-to-frame movement")
    ax.legend()
    return fig


def plot_body_length(rec: Recording, bin_s: float = 10.0) -> plt.Figure:
    """Median body length per fish over time; label swaps between fish of different size
    show up as lines crossing."""
    from electric_fish.tracking import body_length, pose_array

    xy, _, fish, kps = pose_array(rec)
    length = pd.DataFrame(body_length(xy, kps), columns=fish, index=rec.poses.index)
    t = rec.eods.groupby("frame")["time_s"].first().reindex(length.index)
    binned = length.groupby((t // bin_s) * bin_s).median()
    fig, ax = plt.subplots(figsize=(11, 3))
    for f, color in fish_colors(fish).items():
        ax.plot(binned.index, binned[f], color=color, lw=1.5, label=f"fish {f}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("body length (px)")
    ax.set_title(f"Body length, head–middle–tail (median per {bin_s:g} s)")
    ax.legend(loc="upper left", ncols=len(fish))
    return fig
