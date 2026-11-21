"""Side-by-side video, pose tracks and EOD data in a fastplotlib window.

    uv run --group viz python scripts/video_viewer.py [RECORDING]

Drag the yellow cursor in the bottom timeline to scrub through the session: the video frame,
every fish's skeleton, and the voltage of the EOD nearest that frame (10 electrodes, stacked)
follow it. Scroll to zoom and drag to pan inside any panel.

Rendering uses wgpu. On WSL2 without a D3D12 Vulkan driver it falls back to the CPU
(llvmpipe), which manages roughly 25 frames/s for this layout.

For automated checks: ``--offscreen OUT.png [FRAME]`` renders a single frame to a file.
"""

from __future__ import annotations

import argparse

import fastplotlib as fpl
import numpy as np

from electric_fish import load_recording
from electric_fish.eod import peak_to_peak
from electric_fish.plotting import CATEGORICAL
from electric_fish.tracking import pose_array

TIMELINE_BIN = 25  # video frames per bin of the amplitude envelope (0.5 s at 50 fps)


class VideoViewer:
    def __init__(self, recording: str, canvas: str | None = None):
        self.rec = rec = load_recording(recording)
        self.video = rec.open_video()
        self.video.gray = True
        self.xy, _, self.fish, self.kps = pose_array(rec)
        self.pose_frames = rec.poses.index.to_numpy()
        self.eod_frames = rec.eods["frame"].to_numpy()
        amp = np.nanmax(peak_to_peak(rec.waveforms), axis=1)
        n_video = len(self.video)

        self.fig = fig = fpl.Figure(
            rects=[(0, 0, 0.55, 0.72), (0.55, 0, 0.45, 0.72), (0, 0.72, 1, 0.28)],
            names=["video", "EOD voltage (electrodes 1-10, stacked)", "EOD amplitude"],
            size=(1400, 900),
            canvas=canvas,
        )
        self.video_plot = fig["video"]
        first = int(self.pose_frames[0])

        self.image = self.video_plot.add_image(self.video[first], cmap="gray")
        self.skeletons = []
        for a in range(len(self.fish)):
            pts = self._points(0, a)
            line = self.video_plot.add_line(pts, colors=CATEGORICAL[a], thickness=3)
            dots = self.video_plot.add_scatter(pts, colors=CATEGORICAL[a], sizes=8)
            self.skeletons.append((line, dots))

        wave_plot = fig["EOD voltage (electrodes 1-10, stacked)"]
        w = np.asarray(rec.waveforms[0])
        self.waves = wave_plot.add_line_stack(
            np.stack([np.c_[np.arange(w.shape[1]), w[e]] for e in range(w.shape[0])]),
            colors="w",
            separation=(0, 0.6, 0),
        )

        # Amplitude of every EOD (dots) plus a max-per-bin envelope that carries the cursor.
        timeline = fig["EOD amplitude"]
        timeline.add_scatter(np.c_[self.eod_frames, amp], sizes=2, colors="#6da7ec")
        bins = np.arange(0, n_video + TIMELINE_BIN, TIMELINE_BIN)
        envelope = np.zeros(len(bins) - 1)
        np.maximum.at(envelope, np.digitize(self.eod_frames, bins) - 1, np.nan_to_num(amp))
        line = timeline.add_line(np.c_[bins[:-1], envelope], colors="#c3c2b7", thickness=1)
        self.cursor = line.add_linear_selector(selection=first)
        self.cursor.add_event_handler(lambda ev: self.set_frame(ev.info["value"]), "selection")

    def _points(self, row: int, fish_idx: int) -> np.ndarray:
        xy = self.xy[row, fish_idx]
        return np.c_[xy, np.zeros(len(xy))].astype(np.float32)

    def set_frame(self, frame: float) -> None:
        frame = int(np.clip(round(frame), 0, len(self.video) - 1))
        self.image.data = self.video[frame]
        row = min(np.searchsorted(self.pose_frames, frame), len(self.pose_frames) - 1)
        for a, (line, dots) in enumerate(self.skeletons):
            pts = self._points(row, a)
            line.data[:, :3] = pts
            dots.data[:, :3] = pts
        eod = min(np.searchsorted(self.eod_frames, frame), len(self.eod_frames) - 1)
        w = np.asarray(self.rec.waveforms[eod])
        for e, graphic in enumerate(self.waves.graphics):
            graphic.data[:, 1] = w[e]

    def show(self) -> None:
        self.fig.show(maintain_aspect=False)
        self.video_plot.camera.local.scale_y = -1  # image coordinates: y points down
        self.video_plot.auto_scale(maintain_aspect=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recording", nargs="?", default="2024-10-15T14_12_25")
    parser.add_argument("--offscreen", nargs="+", metavar=("OUT.png", "FRAME"))
    args = parser.parse_args()

    if args.offscreen:
        import imageio.v3 as iio

        viewer = VideoViewer(args.recording, canvas="offscreen")
        viewer.show()
        if len(args.offscreen) > 1:
            viewer.cursor.selection = int(args.offscreen[1])
        iio.imwrite(args.offscreen[0], np.asarray(viewer.fig.canvas.draw()))
        return

    viewer = VideoViewer(args.recording)
    viewer.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
