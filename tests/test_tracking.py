import numpy as np
import pandas as pd

from electric_fish import Recording
from electric_fish.tracking import detect_events, qc_summary, size_order_mismatch

KPS = ["mouth", "head", "middle", "tail"]
OFFSETS = {"mouth": 0.0, "head": 20.0, "middle": 70.0, "tail": 140.0}  # along x, px


def make_recording(n=40, lengths=(1.0, 0.6), positions=None):
    """Two straight fish swimming slowly in +y, far apart; returns (recording, xy array)."""
    frames = np.arange(1, n + 1)
    if positions is None:
        positions = np.stack(
            [np.stack([np.full(n, 100.0), 100 + 2.0 * np.arange(n)], 1),
             np.stack([np.full(n, 600.0), 100 + 2.0 * np.arange(n)], 1)], axis=1
        )  # fmt: skip
    cols, data = [], []
    for a, fish in enumerate(["1", "2"]):
        for kp in KPS:
            x = positions[:, a, 0] + OFFSETS[kp] * lengths[a]
            y = positions[:, a, 1]
            for field, v in (("x", x), ("y", y), ("score", np.full(n, 0.9))):
                cols.append((fish, kp, field))
                data.append(v)
    poses = pd.DataFrame(np.array(data).T, index=pd.Index(frames, name="frame"))
    poses.columns = pd.MultiIndex.from_tuples(cols, names=["fish", "keypoint", "field"])
    poses = poses.sort_index(axis=1)
    eods = pd.DataFrame({"frame": frames, "time_s": frames / 50.0})
    rec = Recording("synthetic", "legacy", eods, np.zeros((n, 10, 25), np.float32), poses)
    return rec, positions


def test_clean_tracks_have_no_events():
    rec, _ = make_recording()
    assert detect_events(rec).empty
    assert np.nansum(size_order_mismatch(rec)) == 0
    summary = qc_summary(rec)
    assert summary["swaps"].sum() == summary["jumps"].sum() == 0


def test_swap_detected_once_per_fish():
    rec, pos = make_recording()
    swapped = pos.copy()
    swapped[20:] = swapped[20:, ::-1]  # labels swap from row 20 on
    rec, _ = make_recording(positions=swapped, lengths=(1.0, 1.0))
    ev = detect_events(rec)
    swaps = ev[ev.kind == "swap"]
    assert sorted(swaps.fish) == ["1", "2"]
    assert set(swaps.frame) == {21}
    assert set(swaps.partner) == {"1", "2"}
    assert (ev.kind == "jump").sum() == 0  # explained by the swap


def test_size_order_mismatch_when_whole_fish_swap_labels():
    rec, _ = make_recording()  # fish 1 is long, fish 2 short
    p = rec.poses
    one, two = p["1"].copy(), p["2"].copy()
    p.loc[31:, "1"] = two.loc[31:].to_numpy()  # from frame 31 the labels carry the other body
    p.loc[31:, "2"] = one.loc[31:].to_numpy()
    flags = size_order_mismatch(rec)
    assert flags.loc[:30].sum() == 0
    assert flags.loc[31:].all()


def test_jump_and_length_outlier():
    rec, pos = make_recording()
    jumped = pos.copy()
    jumped[25:, 0, 1] += 300  # fish 1 teleports 300 px
    rec, _ = make_recording(positions=jumped)
    ev = detect_events(rec)
    jumps = ev[ev.kind == "jump"]
    assert list(jumps.fish) == ["1"] and list(jumps.frame) == [26]

    rec.poses.loc[10, ("2", "tail", "x")] += 200  # stretch fish 2 at frame 10
    ev = detect_events(rec)
    lengths = ev[ev.kind == "length"]
    assert list(lengths.fish) == ["2"] and list(lengths.frame) == [10]


def test_large_frame_gaps_are_not_compared():
    rec, pos = make_recording()
    jumped = pos.copy()
    jumped[25:, 0, 1] += 300
    rec, _ = make_recording(positions=jumped)
    rec.poses.index = pd.Index(np.r_[np.arange(1, 26), np.arange(26, 41) + 100], name="frame")
    assert detect_events(rec, max_gap=4).query("kind == 'jump'").empty
