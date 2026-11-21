import numpy as np
import pandas as pd
import pytest

from electric_fish import Recording, find_recordings
from electric_fish.dataset import DATA_DIR
from electric_fish.recording import N_ELECTRODES, N_SAMPLES, recording_timestamp

KEYPOINTS = ["mouth", "head", "tail"]


def _pose_cols(fish):
    return [f"{kp}_{f}_{fi}" for fi in fish for kp in KEYPOINTS for f in ("x", "y", "score")]


def _legacy_df():
    cols = ["Frame", "Time", *_pose_cols(["1", "2"])]
    cols += [f"electrode_{e}_time_{s}" for e in range(1, 11) for s in range(1, 26)]
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, len(cols)))
    df = pd.DataFrame(data, columns=cols)
    df["Frame"] = [1, 1, 2]
    df.iloc[1, 2:-250] = df.iloc[0, 2:-250]  # same frame -> same pose
    # electrode e, sample s encoded as e*100+s to check reshaping order
    for e in range(1, 11):
        for s in range(1, 26):
            df[f"electrode_{e}_time_{s}"] = e * 100 + s
    return df


def _assigned_df():
    tok = [f"m{k:02d}" for k in range(12, 0, -1)] + ["0"] + [f"p{k:02d}" for k in range(1, 13)]
    cols = ["Frame", "Time", "RelativeTime", "discharging_fish", *_pose_cols(["A", "B"])]
    # deliberately shuffled column order: m/p tokens sort wrong alphabetically
    ecols = [f"electrode_{e}_t_{t}" for e in range(1, 11) for t in tok]
    rows = {c: [0.5, 0.5] for c in cols}
    rows.update(
        Frame=[1, 2],
        Time=["2024-10-15 14:12:25.020", "2024-10-15 14:12:25.040"],
        RelativeTime=[0.02, 0.04],
        discharging_fish=["A", "B"],
    )
    for e in range(1, 11):
        for i, t in enumerate(tok):
            rows[f"electrode_{e}_t_{t}"] = [e * 100 + i] * 2
    return pd.DataFrame(rows)[cols + ecols[::-1]]


@pytest.mark.parametrize(
    "make, fmt, fish", [(_legacy_df, "legacy", ["1", "2"]), (_assigned_df, "assigned", ["A", "B"])]
)
def test_from_dataframe(make, fmt, fish):
    df = make()
    rec = Recording.from_dataframe(df)
    assert rec.format == fmt
    assert rec.fish == fish
    assert sorted(rec.keypoints) == sorted(KEYPOINTS)
    assert rec.waveforms.shape == (len(df), N_ELECTRODES, N_SAMPLES)
    # electrode-major, time-ordered samples
    expected = np.array([[e * 100 + s for s in range(N_SAMPLES)] for e in range(1, 11)])
    offset = 1 if fmt == "legacy" else 0
    np.testing.assert_array_equal(rec.waveforms[0], expected + offset)
    assert {"frame", "time_s"} <= set(rec.eods.columns)
    assert not any(c.startswith("electrode") for c in rec.eods.columns)
    assert list(rec.poses.index) == sorted(df["Frame"].unique())
    assert rec.n_pose_conflicts == 0


def test_cache_roundtrip(tmp_path):
    rec = Recording.from_dataframe(_assigned_df(), name="x")
    back = Recording.load_cache(rec.save(tmp_path))
    pd.testing.assert_frame_equal(rec.eods, back.eods)
    pd.testing.assert_frame_equal(rec.poses, back.poses)
    np.testing.assert_array_equal(rec.waveforms, back.waveforms)


def test_recording_timestamp():
    assert recording_timestamp("Data_2fish_2024-02-13T14_33_21_synch_fix_newest.csv") == (
        "2024-02-13T14_33_21"
    )
    assert recording_timestamp("video2024-02-13T14_33_21.avi") == "2024-02-13T14_33_21"


@pytest.mark.skipif(not any(DATA_DIR.glob("*.csv")), reason="raw data not present")
def test_real_data_pairs_with_videos():
    recs = find_recordings()
    assert recs and all(p["video"] is not None for p in recs.values())
