import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Voltage data viewer")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from electric_fish import find_recordings, load_recording
    from electric_fish import plotting as efp
    from electric_fish.eod import peak_to_peak, strongest_electrode

    efp.use_style()

    def show(fig):
        """Render a matplotlib figure and free it (marimo re-runs cells often)."""
        out = mo.as_html(fig)
        plt.close(fig)
        return out

    return (
        efp,
        find_recordings,
        load_recording,
        mo,
        np,
        pd,
        peak_to_peak,
        show,
        strongest_electrode,
    )


@app.cell
def _(mo):
    mo.md("""
    # Voltage data viewer

    Each EOD (electric organ discharge) is a 25-sample snippet centred on its peak (sample 0)
    on each of the 10 electrodes. Pick a session; every view below updates. The first load of
    a session parses the CSV and caches it in `data/cache/`.
    """)
    return


@app.cell
def _(find_recordings, mo):
    recording = mo.ui.dropdown(
        options=list(find_recordings()),
        value="2024-10-15T14_12_25",
        label="Recording",
    )
    recording
    return (recording,)


@app.cell
def _(load_recording, np, peak_to_peak, recording):
    rec = load_recording(recording.value)
    p2p = peak_to_peak(rec.waveforms)
    largest_eod = int(np.nanargmax(np.nanmax(p2p, axis=1)))
    return largest_eod, p2p, rec


@app.cell
def _(mo, np, rec):
    _duration = rec.eods.time_s.iloc[-1] - rec.eods.time_s.iloc[0]
    _n_nan = int(np.isnan(np.asarray(rec.waveforms)).any(axis=(1, 2)).sum())
    mo.hstack(
        [
            mo.stat(f"{len(rec.eods):,}", label="EODs", caption=f"{_n_nan} with NaN samples"),
            mo.stat(f"{_duration:.0f} s", label="Duration"),
            mo.stat(f"{len(rec.eods) / _duration:.1f} Hz", label="Mean EOD rate"),
            mo.stat(", ".join(rec.fish), label="Fish", caption=f"{rec.format} export"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(mo, np, p2p, pd, rec, strongest_electrode):
    _best = strongest_electrode(rec.waveforms)
    _table = pd.DataFrame(
        {
            "median peak-to-peak": np.nanmedian(p2p, axis=0),
            "95th pct peak-to-peak": np.nanpercentile(p2p, 95, axis=0),
            "strongest for (% of EODs)": np.bincount(_best[_best >= 0], minlength=10)
            / len(_best)
            * 100,
        },
        index=pd.RangeIndex(1, 11, name="electrode"),
    ).round(3)
    mo.accordion({"Per-electrode amplitude": _table})
    return


@app.cell
def _(mo):
    mo.md("""
    ## A single EOD on all electrodes

    The panels share a voltage axis, so relative amplitude across electrodes reflects the
    fish's position. Starts at the largest EOD in the session.
    """)
    return


@app.cell
def _(largest_eod, mo, rec):
    eod = mo.ui.slider(
        start=0,
        stop=len(rec.eods) - 1,
        value=largest_eod,
        label="EOD",
        include_input=True,
        full_width=True,
    )
    eod
    return (eod,)


@app.cell
def _(efp, eod, rec, show):
    show(efp.plot_eod(rec, eod.value))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Consecutive EODs as a heatmap

    Rows are EODs around the selected one, each on its strongest electrode and scaled so the
    peak sample is +1. That removes amplitude and polarity, so rows differ only by shape.
    """)
    return


@app.cell
def _(mo):
    heatmap_count = mo.ui.slider(
        start=50, stop=2000, step=50, value=200, label="EODs shown", show_value=True
    )
    heatmap_electrode = mo.ui.dropdown(
        options={"strongest (normalized)": None} | {f"electrode {e}": e for e in range(1, 11)},
        value="strongest (normalized)",
        label="Electrode",
    )
    mo.hstack([heatmap_count, heatmap_electrode], justify="start")
    return heatmap_count, heatmap_electrode


@app.cell
def _(efp, eod, heatmap_count, heatmap_electrode, rec, show):
    _start = max(eod.value - heatmap_count.value // 2, 0)
    show(
        efp.plot_eod_heatmap(
            rec, start=_start, count=heatmap_count.value, electrode=heatmap_electrode.value
        )
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Over time

    Amplitude per electrode (a band moving across electrodes means the discharging fish moved
    past them) and the EOD rate. Legacy exports (Feb 2024) are not assigned to fish, so the
    rate mixes all fish.
    """)
    return


@app.cell
def _(mo):
    bin_s = mo.ui.slider(
        start=1, stop=60, value=5, label="Time bin (s)", show_value=True
    )
    bin_s
    return (bin_s,)


@app.cell
def _(bin_s, efp, mo, rec, show):
    mo.vstack(
        [
            show(efp.plot_amplitude_over_time(rec, bin_s=bin_s.value)),
            show(efp.plot_eod_rate(rec, bin_s=bin_s.value)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Intervals and waveform shape

    Legacy exports store each EOD's own time, so intervals are exact. The assigned export
    (Oct 2024) only stores the frame time, so its intervals come in 20 ms steps and EODs in the
    same frame give 0. The overlay shows a random sample of EODs on their strongest electrode,
    normalized to peak = +1, with the median in black.
    """)
    return


@app.cell
def _(mo):
    max_ms = mo.ui.slider(
        start=50, stop=1000, step=50, value=250, label="Max interval (ms)", show_value=True
    )
    n_overlay = mo.ui.slider(
        start=50, stop=2000, step=50, value=400, label="EODs overlaid", show_value=True
    )
    mo.hstack([max_ms, n_overlay], justify="start")
    return max_ms, n_overlay


@app.cell
def _(efp, max_ms, mo, n_overlay, rec, show):
    mo.vstack(
        [
            show(efp.plot_interval_histogram(rec, max_ms=max_ms.value)),
            show(efp.plot_waveform_overlay(rec, n=n_overlay.value)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
