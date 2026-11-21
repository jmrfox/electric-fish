import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Tracking quality")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from electric_fish import find_recordings, load_recording, tracking
    from electric_fish import plotting as efp

    efp.use_style()

    def show(fig):
        """Render a matplotlib figure and free it (marimo re-runs cells often)."""
        out = mo.as_html(fig)
        plt.close(fig)
        return out

    return efp, find_recordings, load_recording, mo, np, pd, show, tracking


@app.cell
def _(mo):
    mo.md("""
    # Tracking quality

    Checks the pose tracks in the CSV exports without ground truth:

    - **swap**: relabelling fish between consecutive tracked frames cuts total movement by
      more than the swap margin, so identities likely swapped;
    - **jump**: a fish moves faster than the jump threshold (raw tracker output here peaks
      around 85 px/frame);
    - **length**: body length (head–middle–tail) is far from that fish's median, usually a
      misplaced keypoint;
    - **size-order mismatch**: fish differ in size, so ranking each frame's body lengths should
      reproduce the identities. Only informative when sizes are well separated.

    The CSVs only contain frames with an EOD, so "consecutive" means tracked frames at most
    `max gap` video frames apart.
    """)
    return


@app.cell
def _(mo):
    max_speed = mo.ui.slider(20, 300, step=10, value=100, label="Jump threshold (px/frame)",
                             show_value=True)  # fmt: skip
    swap_margin = mo.ui.slider(10, 300, step=10, value=50, label="Swap margin (px)",
                               show_value=True)  # fmt: skip
    length_tol = mo.ui.slider(0.1, 1.0, step=0.05, value=0.3, label="Length tolerance",
                              show_value=True)  # fmt: skip
    max_gap = mo.ui.slider(1, 10, value=4, label="Max gap (frames)", show_value=True)
    mo.vstack([mo.md("### Thresholds"), mo.hstack([max_speed, swap_margin]),
               mo.hstack([length_tol, max_gap])])  # fmt: skip
    return length_tol, max_gap, max_speed, swap_margin


@app.cell
def _(
    find_recordings,
    length_tol,
    load_recording,
    max_gap,
    max_speed,
    np,
    pd,
    swap_margin,
    tracking,
):
    params = dict(
        max_speed=max_speed.value,
        swap_margin=swap_margin.value,
        max_gap=max_gap.value,
        length_tol=length_tol.value,
    )
    _rows = []
    for _name in find_recordings():
        _rec = load_recording(_name)
        _ev = tracking.detect_events(_rec, **params)
        _swaps = _ev[_ev.kind == "swap"]
        _rows.append(
            {
                "recording": _name,
                "fish": len(_rec.fish),
                "tracked frames": len(_rec.poses),
                "swap transitions": _swaps.frame.nunique(),
                "swaps per 1k frames": 1000 * _swaps.frame.nunique() / len(_rec.poses),
                "jumps": int((_ev.kind == "jump").sum()),
                "length outliers": int((_ev.kind == "length").sum()),
                "size-order mismatch": np.nanmean(tracking.size_order_mismatch(_rec)),
            }
        )
    overview = pd.DataFrame(_rows).set_index("recording")
    return overview, params


@app.cell
def _(mo, overview):
    mo.vstack(
        [
            mo.md("## All sessions"),
            overview.assign(
                **{
                    "swaps per 1k frames": overview["swaps per 1k frames"].map("{:.1f}".format),
                    "size-order mismatch": overview["size-order mismatch"].map("{:.1%}".format),
                }
            ),
        ]
    )
    return


@app.cell
def _(find_recordings, mo):
    recording = mo.ui.dropdown(
        options=list(find_recordings()), value="2024-10-15T14_12_25", label="Recording"
    )
    mo.vstack([mo.md("## One session"), recording])
    return (recording,)


@app.cell
def _(load_recording, params, recording, tracking):
    rec = load_recording(recording.value)
    events = tracking.detect_events(rec, **params)
    with rec.open_video() as _v:
        n_video_frames = len(_v)
    return events, n_video_frames, rec


@app.cell
def _(events, mo, n_video_frames, rec, tracking):
    _summary = tracking.qc_summary(rec, events, n_video_frames)
    mo.vstack(
        [
            _summary.assign(
                **{
                    _col: _summary[_col].map(_fmt.format)
                    for _col, _fmt in {
                        "video coverage": "{:.0%}",
                        "score exactly 1": "{:.0%}",
                        "median speed (px/frame)": "{:.1f}",
                        "median body length (px)": "{:.0f}",
                    }.items()
                }
            ),
            mo.md(
                "*Score exactly 1* is not something a tracker outputs on its own: frames like"
                " that were probably interpolated or hand-edited."
            ),
        ]
    )
    return


@app.cell
def _(efp, events, mo, rec, show):
    mo.vstack(
        [
            show(efp.plot_event_timeline(rec, events)),
            mo.hstack(
                [
                    show(efp.plot_speed_histogram(rec)),
                    mo.md("""
                Movement between consecutive tracked frames. Real swimming forms the main
                bulk; a separate cluster far right of the threshold is label swaps or
                keypoints jumping to another fish.
                """),
                ],
                widths=[3, 2],
            ),
            show(efp.plot_body_length(rec)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Inspect flagged events on video

    Panels show the tracked frame before the event, the event frame, and the next tracked
    frame, with each fish's skeleton drawn on the video. Cropping centres on the flagged fish.
    """)
    return


@app.cell
def _(events, mo):
    kind = mo.ui.dropdown(
        options=["swap", "jump", "length"],
        value="swap" if (events.kind == "swap").any() else "jump",
        label="Event type",
    )
    crop = mo.ui.switch(value=True, label="Crop to fish")
    crop_size = mo.ui.slider(100, 900, step=50, value=350, label="Crop size (px)",
                             show_value=True)  # fmt: skip
    mo.hstack([kind, crop, crop_size], justify="start")
    return crop, crop_size, kind


@app.cell
def _(events, kind, mo):
    selected = events[events.kind == kind.value].reset_index(drop=True)
    event_idx = mo.ui.slider(
        start=0,
        stop=max(len(selected) - 1, 0),
        value=0,
        label=f"{kind.value} event (of {len(selected):,})",
        include_input=True,
        full_width=True,
        disabled=selected.empty,
    )
    event_idx
    return event_idx, selected


@app.cell
def _(crop, crop_size, efp, event_idx, mo, np, rec, selected, show):
    if selected.empty:
        _out = mo.md("No events of this type with the current thresholds.")
    else:
        _ev = selected.iloc[event_idx.value]
        _frames = rec.poses.index
        _pos = _frames.get_loc(_ev.frame)
        _nxt = _frames[_pos + 1] if _pos + 1 < len(_frames) else _ev.frame + 1
        _panels = sorted({int(_ev.prev_frame), int(_ev.frame), int(_nxt)})
        _center = None
        if crop.value:
            _row = rec.poses.loc[_ev.frame, _ev.fish]
            _pts = np.c_[_row.xs("x", level="field"), _row.xs("y", level="field")]
            _center = tuple(np.nanmean(_pts, axis=0))
        with rec.open_video() as _video:
            _fig = efp.plot_tracking_frames(
                rec, _video, _panels, center=_center, size=crop_size.value
            )
        _desc = {
            "swap": f"labels of fish {_ev.fish} and {_ev.partner} swap; relabelling saves "
            f"{_ev.value:.0f} px",
            "jump": f"fish {_ev.fish} moves {_ev.value:.0f} px/frame",
            "length": f"fish {_ev.fish} body length {_ev.value:+.0%} vs its median",
        }[_ev.kind]
        _out = mo.vstack(
            [mo.md(f"**Frame {_ev.frame}** (t = {_ev.time_s:.2f} s): {_desc}"), show(_fig)]
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Frame alignment

    Checks which video frame each CSV frame belongs to: drawn on the right frame, keypoints
    land on the (dark) fish. Lowest intensity wins. On all four sessions the answer is offset
    0, so CSV frame *k* is video index *k*. Reads ~750 video frames, so it runs on demand.
    """)
    return


@app.cell
def _(mo):
    run_alignment = mo.ui.run_button(label="Check alignment for this session")
    run_alignment
    return (run_alignment,)


@app.cell
def _(mo, rec, run_alignment, tracking):
    mo.stop(not run_alignment.value)
    _s = tracking.frame_alignment(rec)
    mo.vstack(
        [
            _s.rename_axis("offset").to_frame().round(1),
            mo.md(f"Best offset: **{_s.idxmin()}**"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
