import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from neurosight.detector import MotionDetector, NeuroSightConfig


def build_config(preset, tile_size):
    if preset == "education":
        return NeuroSightConfig.preset_education(tile_size=tile_size, input_color="bgr")
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


def blend_heatmap(frame, saliency):
    saliency_8bit = (np.clip(saliency, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(saliency_8bit, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0.0)


def draw_hud(frame, frame_count, fps, latency_ms, result):
    annotated = frame.copy()
    global_dx, global_dy = result.global_vector
    mean_confidence = float(np.mean(result.confidence_map))
    mean_saliency = float(np.mean(result.saliency_map))

    lines = [
        f"frames: {frame_count}",
        f"fps: {fps:.1f}",
        f"latency: {latency_ms:.1f} ms",
        f"global: dx={global_dx:.3f}, dy={global_dy:.3f}",
        f"confidence: {mean_confidence:.3f}",
        f"saliency: {mean_saliency:.3f}",
    ]

    y = 24
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return annotated


def build_summary(frame_count, start_time, latencies, saliencies, confidences, global_vectors):
    elapsed = max(time.time() - start_time, 1e-9)
    vectors = np.array(global_vectors, dtype=np.float32) if global_vectors else np.zeros((0, 2))

    return {
        "frames_processed": frame_count,
        "elapsed_seconds": elapsed,
        "wall_fps": frame_count / elapsed,
        "mean_latency_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000.0) if latencies else 0.0,
        "detector_fps": float(1.0 / np.mean(latencies)) if latencies else 0.0,
        "mean_saliency": float(np.mean(saliencies)) if saliencies else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_global_dx": float(np.mean(vectors[:, 0])) if len(vectors) else 0.0,
        "mean_global_dy": float(np.mean(vectors[:, 1])) if len(vectors) else 0.0,
        "max_abs_global_dx": float(np.max(np.abs(vectors[:, 0]))) if len(vectors) else 0.0,
        "max_abs_global_dy": float(np.max(np.abs(vectors[:, 1]))) if len(vectors) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run NeuroSight on a live webcam and save output.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Processing/output width.")
    parser.add_argument("--height", type=int, default=480, help="Processing/output height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument(
        "--preset",
        choices=["edge_fast", "robust", "looming"],
        default="edge_fast",
        help="Detector preset. Use edge_fast for performance validation.",
    )
    parser.add_argument("--tile-size", type=int, default=8, help="Detector tile size.")
    parser.add_argument("--output", default="outputs/webcam_neurosight.mp4", help="Annotated video path.")
    parser.add_argument("--metrics", default="outputs/webcam_metrics.json", help="Metrics JSON path.")
    parser.add_argument("--no-display", action="store_true", help="Save output without opening preview windows.")
    parser.add_argument("--no-write", action="store_true", help="Do not write an output video.")
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Disable display, annotation, and video writing for detector/capture benchmarking.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames. 0 means unlimited.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds. 0 means unlimited.")
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Write raw camera frames while still saving metrics.",
    )
    args = parser.parse_args()
    if args.benchmark_only:
        args.no_display = True
        args.no_write = True
        args.no_annotate = True

    output_path = Path(args.output)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_config(args.preset, args.tile_size)
    detector = MotionDetector(config=config)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    writer = None
    if not args.no_write:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (args.width, args.height))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Failed to open output writer: {output_path}")

    print("Starting NeuroSight Live. Press 'q' to quit.")
    print(f"Preset: {args.preset} | tile_size: {args.tile_size}")
    if writer is not None:
        print(f"Saving annotated video to: {output_path}")
    else:
        print("Video writing disabled.")
    print(f"Saving metrics to: {metrics_path}")

    for _ in range(5):
        cap.read()

    frame_count = 0
    latencies = []
    saliencies = []
    confidences = []
    global_vectors = []
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (args.width, args.height))
            curr_time = time.time()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            process_start = time.time()
            result = detector.update(gray_frame, timestamp=curr_time)
            latency = time.time() - process_start

            output_frame = frame
            if result is not None:
                frame_count += 1
                latencies.append(latency)
                saliencies.append(float(np.mean(result.saliency_map)))
                confidences.append(float(np.mean(result.confidence_map)))
                global_vectors.append(result.global_vector)

                recent_latency = float(np.mean(latencies[-30:]) * 1000.0)
                recent_fps = 1000.0 / recent_latency if recent_latency > 0 else 0.0
                if not args.no_annotate:
                    output_frame = blend_heatmap(frame, result.saliency_map)
                    output_frame = draw_hud(output_frame, frame_count, recent_fps, recent_latency, result)

                if args.max_frames and frame_count >= args.max_frames:
                    break

            if writer is not None:
                writer.write(output_frame)

            if not args.no_display:
                cv2.imshow("NeuroSight Live Output", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.duration and (time.time() - start_time) >= args.duration:
                break

    except KeyboardInterrupt:
        print("\nStopping NeuroSight Live.")

    finally:
        summary = build_summary(
            frame_count,
            start_time,
            latencies,
            saliencies,
            confidences,
            global_vectors,
        )
        summary["preset"] = args.preset
        summary["tile_size"] = args.tile_size
        summary["width"] = args.width
        summary["height"] = args.height
        summary["requested_output_fps"] = args.fps
        summary["video_written"] = writer is not None
        summary["display_enabled"] = not args.no_display
        summary["annotation_enabled"] = not args.no_annotate

        metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

        print("\nSaved:")
        if writer is not None:
            print(f"  video: {output_path}")
        print(f"  metrics: {metrics_path}")
        print("\nSummary:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
