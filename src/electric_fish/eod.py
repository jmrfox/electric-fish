"""Basic measurements on EOD waveform snippets of shape (n_eods, n_electrodes, n_samples)."""

from __future__ import annotations

import numpy as np

from electric_fish.recording import N_SAMPLES

#: Sample offsets relative to the EOD peak (the snippet is centred on the peak).
SAMPLE_OFFSETS = np.arange(N_SAMPLES) - N_SAMPLES // 2


def peak_to_peak(waveforms: np.ndarray) -> np.ndarray:
    """Peak-to-peak amplitude per EOD and electrode, shape (n_eods, n_electrodes)."""
    w = np.asarray(waveforms)
    return np.nanmax(w, axis=-1) - np.nanmin(w, axis=-1)


def strongest_electrode(waveforms: np.ndarray) -> np.ndarray:
    """Index (0-based) of the electrode with the largest peak-to-peak amplitude, per EOD.

    EODs whose snippet is entirely NaN get -1.
    """
    p2p = peak_to_peak(waveforms)
    valid = ~np.isnan(p2p).all(axis=1)
    out = np.full(len(p2p), -1)
    out[valid] = np.nanargmax(p2p[valid], axis=1)
    return out


def strongest_trace(waveforms: np.ndarray, normalize: bool = True) -> np.ndarray:
    """The waveform on each EOD's strongest electrode, shape (n_eods, n_samples).

    With ``normalize=True`` each trace is scaled so its centre (peak) sample is +1. This
    removes amplitude (distance to the electrode) and polarity (fish orientation relative to
    the electrode), leaving only waveform shape. EODs with no valid data are NaN.
    """
    w = np.asarray(waveforms, dtype=np.float64)
    idx = strongest_electrode(w)
    traces = np.full((len(w), w.shape[-1]), np.nan)
    ok = idx >= 0
    traces[ok] = w[np.flatnonzero(ok), idx[ok]]
    if normalize:
        centre = traces[:, w.shape[-1] // 2 : w.shape[-1] // 2 + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            traces = traces / centre
    return traces
