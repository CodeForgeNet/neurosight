"""Regression tests for direction recovery across image speeds and tile phases."""

import numpy as np
import pytest

from benchmarks.run_optical_flow_benchmark import (
    generate_synthetic_motion,
    generate_translating_square,
    summarize_neurosight_vectors,
)
from neurosight.detector import MotionDetector, NeuroSightConfig

DIRECTIONS = ('right', 'left', 'up', 'down')
PHASES = (0, 2, 4, 6)


def _predicted(result) -> str:
    predicted, _, _ = summarize_neurosight_vectors(result.motion_vectors)
    return predicted


def _run_sweep(generator, speeds, **generator_kwargs):
    detector = MotionDetector(config=NeuroSightConfig(tile_size=8))
    failures = []
    for speed in speeds:
        for direction in DIRECTIONS:
            for phase in PHASES:
                frame_prev, frame_curr, _, _ = generator(
                    speed=speed,
                    direction=direction,
                    phase=phase,
                    **generator_kwargs,
                )
                predicted = _predicted(detector.process_pair(frame_prev, frame_curr))
                if predicted != direction:
                    failures.append((speed, direction, phase, predicted))
    return failures


@pytest.mark.parametrize(
    "generator,label",
    [
        (generate_translating_square, "translating square"),
        (generate_synthetic_motion, "drifting grating"),
    ],
)
def test_pair_direction_across_speeds_and_phases(generator, label):
    """The stateless pair path must not fall back to slow-motion-only detection."""
    failures = _run_sweep(generator, speeds=(1, 2, 4, 8, 12, 14))
    assert not failures, f"{label} direction failures: {failures}"


def test_pair_response_does_not_collapse_at_high_speed():
    """Confidence must stay in the same range instead of dipping and rebounding."""
    detector = MotionDetector(config=NeuroSightConfig(tile_size=8))
    response = {}
    for speed in (2, 4, 8, 12):
        magnitudes = []
        for direction in DIRECTIONS:
            for phase in PHASES:
                frame_prev, frame_curr, _, _ = generate_translating_square(
                    speed=speed, direction=direction, phase=phase
                )
                result = detector.process_pair(frame_prev, frame_curr)
                magnitudes.append(float(np.hypot(*result.global_vector)))
        response[speed] = float(np.mean(magnitudes))

    peak = max(response.values())
    assert peak > 0.0, response
    for speed, value in response.items():
        assert value >= 0.5 * peak, f"response dropped at {speed} px/frame: {response}"


def test_pair_highpass_preserves_broad_grating():
    """Broad patterns must survive the spatial high-pass at low speeds."""
    failures = _run_sweep(
        generate_synthetic_motion,
        speeds=(2, 4, 8),
        period=64,
    )
    assert not failures, f"broad-grating direction failures: {failures}"


def test_default_scales_cover_low_and_high_speeds():
    assert NeuroSightConfig().scales == (1, 2, 4, 8)
    assert NeuroSightConfig.preset_edge_fast().scales == (1,)
    assert NeuroSightConfig.preset_looming().scales == (1, 2, 4, 8)
