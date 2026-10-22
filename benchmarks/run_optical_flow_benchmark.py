import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

# Add parent directory to path to import neurosight
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neurosight.detector import MotionDetector

CONFIDENCE_THRESHOLD = 0.01
FARNEBACK_DEAD_ZONE = 0.5
# NeuroSight is expected to recover all cardinal directions on these synthetic
# stimuli; this is a regression threshold, not a general accuracy claim.
EXPECTED_ACCURACY = 100.0

# Speed sweep: the default scales are (1, 2, 4, 8). A 32 px-period grating has
# a hard direction ambiguity at 16 px/frame (half a period), so the pass/fail
# range stops at 14. Sub-tile phases are sampled so a result cannot depend on
# where the stimulus falls relative to the 8 px tile grid.
SWEEP_SPEEDS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16)
SWEEP_PHASES = (0, 2, 4, 6)
SWEEP_PASS_MAX_SPEED = 14
SWEEP_MIN_ACCURACY = 90.0


def generate_synthetic_motion(
    width=256, height=256, speed=4, direction='right', phase=0, period=32
):
    """Generate two frames with a drifting sinusoidal grating."""
    x = np.linspace(0, (width / period) * 2 * np.pi, width)
    y = np.linspace(0, (height / period) * 2 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    phase_offset = phase * 2 * np.pi / period
    phase_shift_x = speed * 2 * np.pi / period
    phase_shift_y = speed * 2 * np.pi / period

    dx, dy = 0, 0
    if direction == 'right':
        dx = speed
        f1 = (np.sin(xx + phase_offset) * 127.5 + 127.5).astype(np.uint8)
        f2 = (np.sin(xx + phase_offset - phase_shift_x) * 127.5 + 127.5).astype(np.uint8)
    elif direction == 'left':
        dx = -speed
        f1 = (np.sin(xx + phase_offset) * 127.5 + 127.5).astype(np.uint8)
        f2 = (np.sin(xx + phase_offset + phase_shift_x) * 127.5 + 127.5).astype(np.uint8)
    elif direction == 'down':
        dy = speed
        f1 = (np.sin(yy + phase_offset) * 127.5 + 127.5).astype(np.uint8)
        f2 = (np.sin(yy + phase_offset - phase_shift_y) * 127.5 + 127.5).astype(np.uint8)
    elif direction == 'up':
        dy = -speed
        f1 = (np.sin(yy + phase_offset) * 127.5 + 127.5).astype(np.uint8)
        f2 = (np.sin(yy + phase_offset + phase_shift_y) * 127.5 + 127.5).astype(np.uint8)

    return f1, f2, (dx, dy), np.ones((height, width), dtype=bool)


def generate_translating_square(
    width=256, height=256, speed=4, direction='right', phase=0
):
    """Generate two frames with a single moving square and a foreground mask."""
    dx, dy = 0, 0
    if direction == 'right':
        dx = speed
    elif direction == 'left':
        dx = -speed
    elif direction == 'down':
        dy = speed
    elif direction == 'up':
        dy = -speed

    f1 = np.zeros((height, width), dtype=np.uint8)
    f2 = np.zeros((height, width), dtype=np.uint8)

    y0, y1 = height // 2 - 24, height // 2 + 24
    x0, x1 = width // 2 - 24, width // 2 + 24
    # Shift the initial position so the sweep covers sub-tile phases.
    if dx:
        x0 += phase
        x1 += phase
    if dy:
        y0 += phase
        y1 += phase

    f1[y0:y1, x0:x1] = 255
    f2[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = 255

    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = True

    return f1, f2, (dx, dy), mask


def direction_from_vector(dx, dy, dead_zone=FARNEBACK_DEAD_ZONE):
    """Map a dense-flow summary vector to one cardinal direction."""
    if max(abs(dx), abs(dy)) < dead_zone:
        return 'none'
    if abs(dx) >= abs(dy):
        return 'right' if dx > 0 else 'left'
    return 'down' if dy > 0 else 'up'


def summarize_neurosight_vectors(vectors):
    """Return the confidence-weighted direction and confidence stats.

    A raw tile count lets many weak, noisy tiles outvote a few strong ones.
    Weighting each tile by its confidence matches the global-vector semantics
    and is more stable when only a small part of the frame is moving.
    """
    valid_vectors = [
        v for v in vectors.values()
        if v['direction'] is not None and v['confidence'] > CONFIDENCE_THRESHOLD
    ]
    if not valid_vectors:
        return 'none', 0, 0.0

    direction_scores = Counter()
    for vector in valid_vectors:
        direction_scores[vector['direction']] += vector['confidence']
    pred_dir = direction_scores.most_common(1)[0][0]
    mean_confidence = float(np.mean([v['confidence'] for v in valid_vectors]))
    return pred_dir, len(valid_vectors), mean_confidence


def run_cardinal_benchmark(detector):
    """Compare NeuroSight and Farneback on the eight core synthetic cases."""
    print("Running Optical Flow Benchmark...")
    print(f"Stimuli: translating square + drifting grating | confidence > {CONFIDENCE_THRESHOLD}")
    print("-" * 50)

    directions = ['right', 'left', 'up', 'down']
    stimuli = [
        ('square', generate_translating_square),
        ('grating', generate_synthetic_motion),
    ]

    total_correct = 0
    total_farneback_correct = 0
    total_tests = len(directions) * len(stimuli)

    for stimulus_name, generator in stimuli:
        print(f"\nStimulus: {stimulus_name}")
        for d in directions:
            f1, f2, gt_vec, mask = generator(direction=d)

            # NeuroSight
            t0 = time.time()
            result = detector.process_pair(f1, f2)
            t_ns = time.time() - t0

            # Farneback (Baseline)
            t0 = time.time()
            flow_cv2 = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            t_cv2 = time.time() - t0

            pred_dir, valid_tile_count, mean_confidence = summarize_neurosight_vectors(
                result.motion_vectors
            )
            is_correct = (pred_dir == d)
            if is_correct:
                total_correct += 1

            # Evaluate optical flow over the moving region for localized stimuli.
            # Full-frame mean is still useful for gratings because the whole frame moves.
            mean_cv2 = np.mean(flow_cv2[mask], axis=0)
            farneback_dir = direction_from_vector(mean_cv2[0], mean_cv2[1])
            farneback_correct = (farneback_dir == d)
            if farneback_correct:
                total_farneback_correct += 1

            print(f"Test Direction: {d.upper():<6} | GT: {str(gt_vec):<8}")
            print(
                f"  NeuroSight : {pred_dir:<6} (Correct: {is_correct}) "
                f"| Tiles: {valid_tile_count:<4} | Mean conf: {mean_confidence:.3f} "
                f"| Global: (dx={result.global_vector[0]:.2f}, dy={result.global_vector[1]:.2f}) "
                f"| Time: {t_ns*1000:.1f}ms"
            )
            print(
                f"  Farneback  : {farneback_dir:<6} (Correct: {farneback_correct}) "
                f"| Masked mean flow: (dx={mean_cv2[0]:.2f}, dy={mean_cv2[1]:.2f}) "
                f"| Time: {t_cv2*1000:.1f}ms\n"
            )

    accuracy = total_correct / total_tests * 100
    farneback_accuracy = total_farneback_correct / total_tests * 100
    print("-" * 50)
    print(f"NeuroSight Accuracy: {accuracy:.1f}% (expected {EXPECTED_ACCURACY:.0f}% on these stimuli)")
    print(f"Farneback Mean-Flow Accuracy: {farneback_accuracy:.1f}%")
    print("-" * 50)

    passed = accuracy >= EXPECTED_ACCURACY
    if not passed:
        print("REGRESSION: NeuroSight missed a cardinal direction on synthetic stimuli.")
    return passed, accuracy, farneback_accuracy


def run_speed_sweep(detector):
    """Check direction recovery across speeds and sub-tile phases."""
    print("\nSpeed sweep (default scales, four sub-tile phases per speed)")
    print(f"{'px/frame':>8}  {'square':>10}  {'grating':>10}")
    print("-" * 34)

    directions = ['right', 'left', 'up', 'down']
    stimuli = [
        ('square', generate_translating_square),
        ('grating', generate_synthetic_motion),
    ]
    results = {}
    total_correct = 0
    total_tests = 0

    for speed in SWEEP_SPEEDS:
        row = {}
        for stimulus_name, generator in stimuli:
            correct = 0
            for direction in directions:
                for phase in SWEEP_PHASES:
                    f1, f2, _, _ = generator(speed=speed, direction=direction, phase=phase)
                    result = detector.process_pair(f1, f2)
                    predicted, _, _ = summarize_neurosight_vectors(result.motion_vectors)
                    correct += int(predicted == direction)
            row[stimulus_name] = correct
            total_correct += correct
            total_tests += len(directions) * len(SWEEP_PHASES)
        results[speed] = row
        print(
            f"{speed:8d}  {row['square']:>6}/{len(directions) * len(SWEEP_PHASES):<3}  "
            f"{row['grating']:>6}/{len(directions) * len(SWEEP_PHASES):<3}"
        )

    print("-" * 34)
    overall = total_correct / total_tests * 100
    print(f"Overall speed-sweep accuracy: {overall:.1f}%")

    per_test = len(directions) * len(SWEEP_PHASES)
    failed = []
    for speed in SWEEP_SPEEDS:
        if speed > SWEEP_PASS_MAX_SPEED:
            continue
        for stimulus_name, correct in results[speed].items():
            accuracy = correct / per_test * 100
            if accuracy < SWEEP_MIN_ACCURACY:
                failed.append(f"{stimulus_name} at {speed} px/frame: {accuracy:.1f}%")

    if failed:
        print("REGRESSION: direction accuracy fell below threshold:")
        for failure in failed:
            print(f"  - {failure}")
        return False, overall

    print(
        f"Pass: every stimulus/speed up to {SWEEP_PASS_MAX_SPEED} px/frame "
        f"is at least {SWEEP_MIN_ACCURACY:.0f}% correct across phases."
    )
    return True, overall


def run_benchmark():
    # Need to adjust the path to weights since we are running from benchmarks/
    weights_path = os.path.join(
        os.path.dirname(__file__), '..', 'neurosight', 'data', 'connectome_weights.json'
    )
    if not os.path.exists(weights_path):
        weights_path = 'connectome_weights.json'  # fallback

    detector = MotionDetector(connectome_weights_path=weights_path, tile_size=8)

    cardinal_passed, accuracy, farneback_accuracy = run_cardinal_benchmark(detector)
    sweep_passed, sweep_accuracy = run_speed_sweep(detector)

    print("-" * 50)
    print(
        f"Summary: cardinal {accuracy:.1f}% vs Farneback {farneback_accuracy:.1f}%; "
        f"speed sweep {sweep_accuracy:.1f}%."
    )
    return int(cardinal_passed and sweep_passed)


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
