"""Quality checks for multi-animal pose tracks.

The checks use only the tracks themselves (no ground truth):

* **jump** - a fish moves faster than ``max_speed`` px/frame between consecutive tracked
  frames. Raw tracker output in these recordings tops out around 85 px/frame (99.9th pct).
* **swap** - re-assigning fish labels between two consecutive tracked frames reduces the
  total keypoint displacement by more than ``swap_margin`` px: identities likely swapped.
* **length** - the fish's body length (head-middle-tail polyline) deviates from that fish's
  median by more than ``length_tol``: keypoints on the wrong body, or a swap between fish of
  different size.
* **size-order mismatch** (per-frame rate, not an event) - fish in a session differ in size,
  so matching each frame's body lengths to the per-fish medians should give the identity
  assignment; frames where it doesn't suggest swapped identities.

Also reported: missing keypoints, and the fraction of scores exactly 1.0, which a tracker
does not produce on its own (it suggests interpolated or hand-edited frames).
"""

from __future__ import annotations

import warnings
from itertools import permutations

import numpy as np
import pandas as pd

from electric_fish.recording import Recording

#: Keypoint order along the body, front to back.
BODY_ORDER = ["chin", "mouth", "head", "middle", "tail"]


def pose_array(rec: Recording) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Poses as arrays.

    Returns ``xy`` with shape (n_frames, n_fish, n_keypoints, 2), ``score`` with shape
    (n_frames, n_fish, n_keypoints), and the fish and keypoint names (keypoints in body order).
    """
    fish = rec.fish
    kps = [k for k in BODY_ORDER if k in rec.keypoints]
    p = rec.poses
    xy = np.stack(
        [np.stack([p[(f, k)][["x", "y"]].to_numpy(float) for k in kps], axis=1) for f in fish],
        axis=1,
    )
    score = np.stack(
        [np.stack([p[(f, k, "score")].to_numpy(float) for k in kps], axis=1) for f in fish],
        axis=1,
    )
    return xy, score, fish, kps


def body_length(xy: np.ndarray, keypoints: list[str]) -> np.ndarray:
    """Head -> middle -> tail polyline length, shape (n_frames, n_fish). NaN if any is missing."""
    idx = [keypoints.index(k) for k in ("head", "middle", "tail")]
    pts = xy[:, :, idx]
    return np.linalg.norm(np.diff(pts, axis=2), axis=-1).sum(axis=2)


def transition_costs(xy: np.ndarray) -> np.ndarray:
    """Mean keypoint distance from fish a at row t to fish b at row t+1.

    Shape (n_frames - 1, n_fish, n_fish); keypoints missing in either frame are skipped.
    """
    d = np.linalg.norm(xy[:-1, :, None] - xy[1:, None, :], axis=-1)  # (T-1, F, F, K)
    with warnings.catch_warnings():  # all-NaN rows -> NaN, handled by callers
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(d, axis=-1)


def detect_events(
    rec: Recording,
    max_speed: float = 100.0,
    swap_margin: float = 50.0,
    max_gap: int = 4,
    length_tol: float = 0.3,
) -> pd.DataFrame:
    """Flag suspicious tracking events.

    Parameters
    ----------
    max_speed
        Displacement per frame (px) above which a move counts as a jump.
    swap_margin
        Total displacement (px, summed over fish) that a relabelling must save to count as
        a swap.
    max_gap
        Only compare tracked frames at most this many video frames apart (the CSVs only
        contain frames with an EOD, so consecutive rows can skip frames).
    length_tol
        Relative deviation from the fish's median body length that counts as an outlier.

    Returns
    -------
    DataFrame with one row per (event, fish): ``frame``, ``prev_frame``, ``time_s``, ``fish``,
    ``kind`` (jump / swap / length), ``value`` (px/frame for jump, px saved for swap, relative
    deviation for length) and ``partner`` (the fish it swapped with).
    """
    xy, _, fish, kps = pose_array(rec)
    frames = rec.poses.index.to_numpy()
    times = rec.eods.groupby("frame")["time_s"].first().reindex(frames).to_numpy()
    gap = np.diff(frames)
    cost = transition_costs(xy)  # (T-1, F, F)
    n_fish = len(fish)
    rows = []

    # Swaps: best relabelling over all permutations (n_fish <= 4 -> at most 24).
    perms = np.array(list(permutations(range(n_fish))))
    fi = np.arange(n_fish)
    perm_cost = cost[:, fi, perms].sum(axis=-1)  # (T-1, P)
    identity_cost = perm_cost[:, 0]
    ok = (gap <= max_gap) & np.isfinite(perm_cost).all(axis=1)
    best = np.where(ok, np.argmin(np.where(np.isfinite(perm_cost), perm_cost, np.inf), axis=1), 0)
    saved = identity_cost - perm_cost[np.arange(len(best)), best]
    is_swap = ok & (best != 0) & (saved > swap_margin)
    swapped_fish = np.zeros((len(gap), n_fish), bool)
    for t in np.flatnonzero(is_swap):
        p = perms[best[t]]
        for a in np.flatnonzero(p != fi):
            swapped_fish[t, a] = True
            rows.append((frames[t + 1], frames[t], times[t + 1], fish[a], "swap", saved[t],
                         fish[p[a]]))  # fmt: skip

    # Jumps (not already explained as a swap).
    speed = np.diagonal(cost, axis1=1, axis2=2) / gap[:, None]  # (T-1, F)
    is_jump = (gap[:, None] <= max_gap) & (speed > max_speed) & ~swapped_fish
    for t, a in zip(*np.nonzero(is_jump), strict=True):
        rows.append((frames[t + 1], frames[t], times[t + 1], fish[a], "jump", speed[t, a], None))

    # Body-length outliers.
    length = body_length(xy, kps)
    rel = length / np.nanmedian(length, axis=0) - 1
    for t, a in zip(*np.nonzero(np.abs(np.nan_to_num(rel)) > length_tol), strict=True):
        rows.append((frames[t], frames[t - 1] if t else frames[t], times[t], fish[a], "length",
                     rel[t, a], None))  # fmt: skip

    events = pd.DataFrame(
        rows, columns=["frame", "prev_frame", "time_s", "fish", "kind", "value", "partner"]
    )
    return events.sort_values(["frame", "fish", "kind"]).reset_index(drop=True)


def size_order_mismatch(rec: Recording) -> pd.Series:
    """Per-frame flag: matching body lengths to per-fish medians does not give the identity.

    Only meaningful when fish differ in size; frames with any missing length are NaN.
    """
    xy, _, fish, kps = pose_array(rec)
    length = body_length(xy, kps)
    ref = np.nanmedian(length, axis=0)
    perms = np.array(list(permutations(range(len(fish)))))
    # cost of assigning observed fish a's length to reference fish perm[a]
    err = np.abs(length[:, None, :] - ref[perms][None]).sum(axis=-1)  # (T, P)
    valid = np.isfinite(length).all(axis=1)
    out = np.where(valid, np.argmin(np.nan_to_num(err, nan=np.inf), axis=1) != 0, np.nan)
    return pd.Series(out, index=rec.poses.index, name="size_order_mismatch")


def qc_summary(rec: Recording, events: pd.DataFrame | None = None, n_video_frames=None):
    """One row per fish with coverage, missing-data and event counts.

    The size-order check is per frame, not per fish: see :func:`size_order_mismatch`.
    """
    if events is None:
        events = detect_events(rec)
    xy, score, fish, kps = pose_array(rec)
    length = body_length(xy, kps)
    gap = np.diff(rec.poses.index.to_numpy())
    speed = np.diagonal(transition_costs(xy), axis1=1, axis2=2)[gap == 1]
    counts = events.groupby(["fish", "kind"]).size().unstack(fill_value=0)
    rows = []
    for a, f in enumerate(fish):
        missing = np.isnan(xy[:, a, :, 0]).mean(axis=0)
        rows.append(
            {
                "fish": f,
                "tracked frames": len(rec.poses),
                "video coverage": len(rec.poses) / n_video_frames if n_video_frames else np.nan,
                "missing (worst keypoint)": f"{missing.max():.1%} ({kps[int(missing.argmax())]})",
                "score exactly 1": np.mean(score[:, a] == 1),
                "median speed (px/frame)": np.nanmedian(speed[:, a]),
                "median body length (px)": np.nanmedian(length[:, a]),
                "swaps": int(counts.get("swap", pd.Series()).get(f, 0)),
                "jumps": int(counts.get("jump", pd.Series()).get(f, 0)),
                "length outliers": int(counts.get("length", pd.Series()).get(f, 0)),
            }
        )
    return pd.DataFrame(rows).set_index("fish")


def frame_alignment(
    rec: Recording, offsets=(-2, -1, 0, 1, 2), n_frames: int = 150, seed: int = 0
) -> pd.Series:
    """Check which video index each CSV frame belongs to.

    Fish are dark on a light background, so when poses are drawn on the right video frame the
    keypoints land on dark pixels. For a random sample of frames where fish are moving (so
    neighbouring frames differ), returns the mean grayscale intensity under the keypoints for
    each candidate ``offset`` (video index = CSV frame + offset). The lowest value wins.
    """
    xy, *_ = pose_array(rec)
    frames = rec.poses.index.to_numpy()
    gap = np.diff(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmax(np.diagonal(transition_costs(xy), axis1=1, axis2=2), axis=1)
    candidates = np.flatnonzero((gap == 1) & (speed > 8) & (speed < 60)) + 1
    rng = np.random.default_rng(seed)
    pick = rng.choice(candidates, size=min(n_frames, len(candidates)), replace=False)
    sums = {o: [] for o in offsets}
    with rec.open_video() as video:
        video.gray = True
        h, w = video.shape
        for t in pick:
            pts = xy[t].reshape(-1, 2)
            pts = pts[np.isfinite(pts).all(axis=1)].round().astype(int)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
            if not len(pts):
                continue
            for o in offsets:
                i = frames[t] + o
                if 0 <= i < len(video):
                    sums[o].append(video[i][pts[:, 1], pts[:, 0]].astype(float).mean())
    return pd.Series({o: np.mean(v) for o, v in sums.items()}, name="keypoint intensity")
