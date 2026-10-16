"""Regression tests for the video pipeline and weight loading.

Covers:
- process_video always returns a saliency ndarray (never None)
- output video inherits the source container frame rate
- timestamp dt tracks the container FPS
- VideoProcessor.probe_video metadata
- connectome weight validation / fallback behaviour
"""

import json

import cv2
import numpy as np
import pytest

from neurosight.detector import MotionDetector
from neurosight.neurosight_api import NeuroSightDetector
from neurosight.utils import VideoProcessor, VideoProperties


def _write_test_video(path, width=64, height=48, frames=12, fps=10.0):
    """Write a short video with a moving bright square."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "test video writer failed to open"
    try:
        for t in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 4 + (t * 3) % (width - 16)
            frame[16:32, x:x + 12] = 255
            writer.write(frame)
    finally:
        writer.release()


def test_probe_video_reports_metadata(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_test_video(video, width=80, height=60, fps=15.0)

    props = VideoProcessor().probe_video(str(video))

    assert isinstance(props, VideoProperties)
    assert props.width == 80
    assert props.height == 60
    assert props.fps == pytest.approx(15.0, abs=0.5)
    assert props.duration > 0
    assert props.frame_interval == pytest.approx(1.0 / props.fps, rel=1e-3)


def test_process_video_returns_array_and_preserves_fps(tmp_path):
    video = tmp_path / "clip.mp4"
    output = tmp_path / "annotated.mp4"
    _write_test_video(video, fps=12.0, frames=20)

    detector = NeuroSightDetector(tile_size=8, resize_width=None)
    saliency, vectors = detector.process_video(
        str(video), output_video=str(output), verbose=False
    )

    # #1: never returns None; shape matches the source frame.
    assert isinstance(saliency, np.ndarray)
    assert saliency.ndim == 2
    assert saliency.shape == (48, 64)
    assert len(vectors) > 0

    # #2: the annotated output must play at the source frame rate.
    written = VideoProcessor().probe_video(str(output))
    assert written.fps == pytest.approx(12.0, abs=0.5)


def test_single_frame_video_returns_zero_saliency(tmp_path):
    video = tmp_path / "one_frame.mp4"
    _write_test_video(video, frames=1)

    detector = NeuroSightDetector(tile_size=8, resize_width=None)
    saliency, vectors = detector.process_video(str(video), verbose=False)

    assert isinstance(saliency, np.ndarray)
    assert saliency.shape == (48, 64)
    assert float(saliency.max()) == 0.0
    assert vectors == {}


def test_load_weights_malformed_json_raises(tmp_path):
    bad = tmp_path / "weights.json"
    bad.write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        MotionDetector(connectome_weights_path=str(bad))


def test_load_weights_missing_file_falls_back_to_bundle():
    # A non-existent path must fall back to the bundled weights, not crash.
    detector = MotionDetector(connectome_weights_path="/does/not/exist.json")
    assert detector.weights
    assert "t4_a" in detector.weights


def test_load_weights_rejects_non_finite(tmp_path):
    bad = tmp_path / "weights.json"
    bad.write_text(json.dumps({"weights": {"t4_a": [0.5, float("inf")]}}))
    with pytest.raises(ValueError, match="non-finite"):
        MotionDetector(connectome_weights_path=str(bad))


def test_read_video_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        list(VideoProcessor().read_video(str(tmp_path / "missing.mp4")))


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        VideoProcessor().probe_video(str(tmp_path / "missing.mp4"))


def test_performance_monitor_reports():
    from neurosight.utils import PerformanceMonitor

    monitor = PerformanceMonitor()
    assert monitor.get_fps() == 0.0
    monitor.record_frame(0.01)
    monitor.record_frame(0.02)
    assert monitor.frame_count == 2
    assert monitor.get_latency_ms() > 0
    assert "Frames: 2" in monitor.report()


def test_reset_clears_adapted_threshold():
    detector = MotionDetector()
    detector.update(np.full((32, 32), 200, dtype=np.uint8))
    detector.update(np.full((32, 32), 20, dtype=np.uint8))
    # Adaptation moves the threshold away from the default.
    assert detector.sigmoid_threshold != detector.DEFAULT_SIGMOID_THRESHOLD

    detector.reset()
    assert detector.sigmoid_threshold == detector.DEFAULT_SIGMOID_THRESHOLD
