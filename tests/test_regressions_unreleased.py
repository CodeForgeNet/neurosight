"""Regression tests for the Unreleased production-readiness fixes."""

import numpy as np
import pytest

from neurosight.detector import MotionDetector, NeuroSightConfig
from neurosight.neurosight_api import NeuroSightDetector


def _frame(seed: int, shape=(64, 64)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def test_nan_frame_is_rejected_and_does_not_poison_threshold():
    detector = MotionDetector(config=NeuroSightConfig(tile_size=8))
    detector.update(_frame(0))
    with pytest.raises(ValueError, match="NaN or Inf"):
        detector.update(np.full((64, 64), np.nan, dtype=np.float32))
    assert np.isfinite(detector.sigmoid_threshold)


def test_inf_and_empty_frames_are_rejected():
    detector = MotionDetector(config=NeuroSightConfig(tile_size=8))
    with pytest.raises(ValueError):
        detector.update(np.full((8, 8), np.inf, dtype=np.float32))
    with pytest.raises(ValueError):
        detector.update(np.zeros((0, 8), dtype=np.float32))


def test_float_frames_above_255_are_rejected():
    detector = MotionDetector(config=NeuroSightConfig(tile_size=8))
    with pytest.raises(ValueError, match="values above 255"):
        detector.update(np.full((8, 8), 300.0, dtype=np.float32))


def test_process_pair_is_independent_of_streaming_history():
    prev = _frame(1)
    curr = np.roll(prev, 1, axis=1)
    config = NeuroSightConfig(tile_size=8)

    fresh = MotionDetector(config=config).process_pair(prev, curr).saliency_map

    used = MotionDetector(config=config)
    for seed in range(2, 52):
        used.update(_frame(seed))
    after_history = used.process_pair(prev, curr).saliency_map

    np.testing.assert_allclose(fresh, after_history)


def test_process_video_does_not_leak_state_between_videos(tmp_path):
    from tests.test_video_pipeline import _write_test_video

    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    _write_test_video(first, fps=12.0, frames=10)
    _write_test_video(second, fps=12.0, frames=10)

    fresh_saliency, _ = NeuroSightDetector(tile_size=8, resize_width=None).process_video(
        str(second), verbose=False
    )

    reused = NeuroSightDetector(tile_size=8, resize_width=None)
    reused.process_video(str(first), verbose=False)
    reused_saliency, _ = reused.process_video(str(second), verbose=False)

    # If state leaked, the first frames of `second` would be compared against
    # the last frames of `first`, and the saliency would differ from a fresh run.
    np.testing.assert_allclose(reused_saliency, fresh_saliency)


def test_resolution_change_with_temporal_denoising_resets_cleanly():
    detector = MotionDetector(
        config=NeuroSightConfig(tile_size=8, temporal_denoising=True)
    )
    assert detector.update(np.zeros((32, 48), dtype=np.uint8)) is None

    # A shape change must reset the smoothed-frame state before blending, not
    # attempt to broadcast arrays of different sizes.
    assert detector.update(np.zeros((48, 64), dtype=np.uint8)) is None
    assert detector._last_frame_shape == (48, 64)

    result = detector.update(np.zeros((48, 64), dtype=np.uint8))
    assert result is not None
    assert result.diagnostics['frame_shape'] == (48, 64)


def test_process_video_omits_vectors_by_default(tmp_path):
    from tests.test_video_pipeline import _write_test_video

    video = tmp_path / "clip.mp4"
    _write_test_video(video, fps=12.0, frames=10)

    detector = NeuroSightDetector(tile_size=8, resize_width=None)
    _, vectors = detector.process_video(str(video), verbose=False)
    assert vectors == {}


def test_edge_fast_preset_disables_robotics_outputs():
    config = NeuroSightConfig.preset_edge_fast()
    assert config.enable_regions is False
    assert config.enable_steering is False
    assert config.enable_looming is False
    assert config.include_pathway_maps is False


def test_robotics_outputs_are_disabled_by_default():
    config = NeuroSightConfig()
    assert config.enable_regions is False
    assert config.enable_looming is False
    assert config.enable_steering is False

    detector = MotionDetector(config=config)
    frame = np.zeros((32, 32), dtype=np.uint8)
    frame[10:20, 10:20] = 255
    detector.update(frame)
    result = detector.update(frame)
    assert result.looming_score == 0.0
    assert result.time_to_contact is None
    assert result.regions == []
    assert result.steering_signal == {
        'turn': 0.0,
        'forward_caution': 0.0,
        'collision_urgency': 0.0,
    }


def test_streaming_direction_across_speeds_and_tile_phases():
    """Displacements up to 3 px/frame must not depend on tile phase."""
    from collections import Counter

    def dominant(result):
        scores = Counter()
        for vector in result.motion_vectors.values():
            if vector['direction'] is not None and vector['confidence'] > 0.01:
                scores[vector['direction']] += vector['confidence']
        return scores.most_common(1)[0][0] if scores else 'none'

    def sequence(direction, speed, phase, size=96, frames=8):
        dx_step = {'right': 1, 'left': -1}.get(direction, 0) * speed
        dy_step = {'down': 1, 'up': -1}.get(direction, 0) * speed
        out = []
        for t in range(frames):
            frame = np.zeros((size, size), dtype=np.uint8)
            y = 24 + dy_step * t
            x = 24 + phase + dx_step * t
            frame[y:y + 12, x:x + 12] = 255
            out.append(frame)
        return out

    for speed in (1, 2, 3, 4):
        for direction in ('right', 'left', 'up', 'down'):
            for phase in (0, 2, 4, 6):
                frames = sequence(direction, speed, phase)
                detector = MotionDetector(tile_size=8)
                for frame in frames[:-1]:
                    detector.update(frame)
                predicted = dominant(detector.update(frames[-1]))
                assert predicted == direction, (
                    f"speed={speed} direction={direction} phase={phase}: "
                    f"predicted {predicted}"
                )


def test_motion_near_frame_border_is_detected():
    frames = []
    for t in range(8):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[24:36, 48 + t * 2:60 + t * 2] = 255
        frames.append(frame)

    detector = MotionDetector(tile_size=8)
    for frame in frames[:-1]:
        detector.update(frame)
    result = detector.update(frames[-1])

    assert result.diagnostics['active_tiles'] > 0
    assert result.global_vector[0] > 0


def test_static_textured_pair_has_no_active_tiles():
    frame = (np.random.default_rng(7).random((64, 64)) * 255).astype(np.uint8)
    result = MotionDetector(tile_size=8).process_pair(frame, frame)

    assert result.diagnostics['active_tiles'] == 0
    assert np.max(result.magnitude_map) == 0.0
    assert np.max(result.saliency_map) == 0.0


def test_end_to_end_expansion_triggers_looming_but_translation_does_not():
    import cv2

    size = 64
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (5, 5), 0)

    expansion = []
    for t in range(8):
        crop = int(size * (4 - t * 0.3))
        y0 = (texture.shape[0] - crop) // 2
        x0 = (texture.shape[1] - crop) // 2
        frame = cv2.resize(
            texture[y0:y0 + crop, x0:x0 + crop], (size, size), interpolation=cv2.INTER_LINEAR
        )
        expansion.append((frame * 255).astype(np.uint8))

    translation = []
    for t in range(8):
        frame = np.zeros((size, size), dtype=np.uint8)
        frame[20:44, 4 + t * 4:28 + t * 4] = 200
        translation.append(frame)

    def run(frames):
        detector = MotionDetector(config=NeuroSightConfig.preset_looming(tile_size=8))
        for frame in frames[:-1]:
            detector.update(frame)
        return detector.update(frames[-1])

    expansion_result = run(expansion)
    translation_result = run(translation)

    assert expansion_result.looming_score > 0.05
    assert expansion_result.time_to_contact is not None
    assert expansion_result.time_to_contact > 0.0
    assert translation_result.looming_score < 0.05
