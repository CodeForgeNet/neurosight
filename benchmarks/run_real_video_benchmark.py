import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from neurosight.detector import MotionDetector, NeuroSightConfig


def build_config(preset, tile_size):
    if preset == "edge_fast":
        return NeuroSightConfig.preset_edge_fast(
            tile_size=tile_size,
            input_color="bgr",
            enable_regions=False,
            enable_steering=False,
        )
    if preset == "robust":
        return NeuroSightConfig.preset_robust(tile_size=tile_size, input_color="bgr")
    if preset == "looming":
        return NeuroSightConfig.preset_looming(tile_size=tile_size, input_color="bgr")
    raise ValueError(f"unsupported preset: {preset}")


def read_frames(video_path, width, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot read video: {video_path}")

    frames = []
    try:
        while max_frames <= 0 or len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if width:
                h, w = frame.shape[:2]
                height = int(h * (width / w))
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()

    if len(frames) < 2:
        raise ValueError("Need at least two readable frames for benchmarking.")
    return frames


def benchmark(frames, preset, tile_size):
    detector = MotionDetector(config=build_config(preset, tile_size))
    rows = []
    saliencies = []
    confidences = []
    global_vectors = []

    for index, frame in enumerate(frames):
        start = time.perf_counter()
        result = detector.update(frame, timestamp=index / 30.0)
        elapsed = time.perf_counter() - start

        if result is None:
            continue

        rows.append({"frame": index, "latency_ms": elapsed * 1000.0})
        saliencies.append(float(np.mean(result.saliency_map)))
        confidences.append(float(np.mean(result.confidence_map)))
        global_vectors.append(result.global_vector)

    latencies = np.array([row["latency_ms"] for row in rows], dtype=np.float32)
    vectors = np.array(global_vectors, dtype=np.float32) if global_vectors else np.zeros((0, 2))

    return {
        "preset": preset,
        "tile_size": tile_size,
        "frames_processed": len(rows),
        "mean_latency_ms": float(np.mean(latencies)) if len(latencies) else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        "detector_fps": float(1000.0 / np.mean(latencies)) if len(latencies) else 0.0,
        "mean_saliency": float(np.mean(saliencies)) if saliencies else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_global_dx": float(np.mean(vectors[:, 0])) if len(vectors) else 0.0,
        "mean_global_dy": float(np.mean(vectors[:, 1])) if len(vectors) else 0.0,
    }, rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark NeuroSight on a real video file.")
    parser.add_argument("video", help="Path to input video.")
    parser.add_argument(
        "--preset",
        choices=["edge_fast", "robust", "looming"],
        default="edge_fast",
        help="Detector preset to benchmark.",
    )
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=640, help="Resize input width before processing.")
    parser.add_argument("--max-frames", type=int, default=300, help="0 means all frames.")
    parser.add_argument("--csv", default=None, help="Optional per-frame latency CSV path.")
    args = parser.parse_args()

    frames = read_frames(Path(args.video), args.width, args.max_frames)
    summary, rows = benchmark(frames, args.preset, args.tile_size)

    print("Real Video Benchmark")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "latency_ms"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
