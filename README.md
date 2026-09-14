# NeuroSight

[![PyPI version](https://badge.fury.io/py/neurosight.svg)](https://badge.fury.io/py/neurosight)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/CodeForgeNet/neurosight/actions/workflows/build.yml/badge.svg)](https://github.com/CodeForgeNet/neurosight/actions)

**Lightweight motion detection inspired by the fly optic lobe.**

![NeuroSight Demo](assets/demo.gif)

A lightweight, CPU-friendly motion detection library inspired by the *Drosophila* (fruit fly) visual system. NeuroSight implements a Hassenstein-Reichardt-style correlator and uses extracted T4/T5 connectome weights to produce directional motion saliency from ordinary video frames.

NeuroSight is not intended to replace full optical flow or deep models such as RAFT. Its niche is small, inspectable, biologically inspired motion detection for education, experiments, and edge-device prototypes.

## Why NeuroSight?

- **Biologically inspired**: Uses a fly-vision motion detector model with T4/T5 direction-selective weighting
- **Lightweight**: Small Python package, CPU-only, no neural-network checkpoint or GPU runtime
- **Inspectable**: The core algorithm is NumPy-based and easy to read, tune, and teach
- **Self-contained at runtime**: Ships with pre-extracted weights, so no external neuroscience API calls are required during use

### Who is this for?

- **Robotics & Edge-AI Builders**: Need simple directional motion cues on CPU-only devices.
- **Students & Researchers**: Want a compact implementation of fly-inspired motion detection.
- **Neuroscience / Computer Vision Educators**: Need a practical demo of correlator-based motion saliency.
- **Hobbyists**: Want to experiment with motion detection on webcams, Raspberry Pi-style devices, or small robots.

## Installation

### From PyPI
```bash
pip install neurosight
```

### From source
```bash
git clone https://github.com/CodeForgeNet/neurosight
cd neurosight
pip install -e .
```

## How to Use It

NeuroSight ships with pre-extracted biological connectome weights built directly into the library, so you can start detecting motion immediately.

### Quick Example: Process a Video

The simplest way to use NeuroSight is to process a video file and output an annotated video with a motion heatmap and directional arrows.

```python
from neurosight import NeuroSightDetector

detector = NeuroSightDetector()

saliency, motion_vectors = detector.process_video(
    "input.mp4",
    output_video="output_with_motion.mp4",
    overlay_vectors=True
)
```

### Real-time / Frame-by-Frame Processing

If you are reading from a webcam or integrating into a larger robotics pipeline, you can process individual frame pairs in real time:

```python
import cv2
from neurosight import NeuroSightDetector

detector = NeuroSightDetector()
cap = cv2.VideoCapture(0)

ret, frame_prev = cap.read()

while True:
    ret, frame_curr = cap.read()
    if not ret: break
    
    # Detect motion between two frames
    saliency, vectors = detector.process_frame_pair(frame_prev, frame_curr)
    
    # Display the result
    cv2.imshow("Motion Saliency", (saliency * 255).astype('uint8'))
    if cv2.waitKey(1) == ord('q'): break
    
    frame_prev = frame_curr
```

## What it Provides

### Saliency Map
The library returns a `saliency` map, which is a 2D NumPy array representing the intensity of motion detected across the visual field (scaled from `0.0` to `1.0`).

### Motion Vectors
For more precise data, the library returns a dictionary of motion vectors broken down into local spatial tiles (e.g., 8x8 pixels). Each tile contains:
- `direction`: The dominant motion direction (`up`, `down`, `left`, `right`)
- `confidence`: The strength of the detection (`0.0 - 1.0`)
- `x`, `y`: The tile coordinates

## Architecture & Biological Background

Fly-inspired motion detection is an established research area. NeuroSight packages one practical version of that idea as a small Python library, combining a Hassenstein-Reichardt-style correlator with pre-extracted T4/T5 weighting:

- **T4/T5 Neurons**: The core of fly motion vision. T4 neurons detect moving bright edges (ON pathway), and T5 neurons detect moving dark edges (OFF pathway). Each is divided into 4 subtypes corresponding to the 4 cardinal directions (up, down, left, right).
- **Medulla Interneurons**: Signals from photoreceptors are split into L1 (light-sensitive) and L2 (dark-sensitive) channels. They pass through medulla interneurons (Mi1, Tm3, etc.) which act as temporal delay lines.
- **Hassenstein-Reichardt Correlator**: By multiplying a delayed signal from one photoreceptor with an undelayed signal from a neighboring one, the brain determines direction and speed.
- **Connectome Weights**: The included weights are derived from *Drosophila* hemibrain connectivity and normalized for this implementation.

## Performance

Indicative benchmark on Intel i7-12700K, 1280×720 video:

| Metric | NeuroSight | OpenCV OpticalFlow | TensorFlow RAFT |
|--------|-----------|-------------------|-----------------|
| FPS | 45-50 | 30-35 | 5-10 (GPU) |
| Latency (ms) | 20-25 | 28-35 | 100+ |
| Memory (MB) | 15 | 25 | 500+ |
| Model size | <1MB | N/A | 200MB |

These figures are intended as rough positioning, not a claim of universal accuracy or speed. Real performance depends on frame size, tile size, hardware, OpenCV build, and the scene being processed.

### Directional Synthetic Benchmark

NeuroSight also includes a small directional benchmark using a drifting sinusoidal grating (`period=32px`, `speed=4px/frame`) across the four cardinal directions:

```bash
python benchmarks/run_optical_flow_benchmark.py
```

| Metric | NeuroSight | Farneback mean-flow baseline |
|--------|-----------:|-----------------------------:|
| Direction accuracy | 100.0% | 0.0% |
| Valid NeuroSight tiles | 1024 / 1024 | N/A |
| Mean NeuroSight confidence | 0.258 | N/A |

This benchmark tests direction recovery on a periodic grating. Farneback is summarized by the mean dense-flow vector, which can collapse toward zero on this stimulus with the current parameters. Treat this as a scoped synthetic benchmark, not a general claim that classical optical flow fails.

## FAQ

**Q: Is this better than optical flow?**  
A: Not generally. NeuroSight is complementary: it provides lightweight directional motion saliency, while classical and deep optical-flow methods are usually better when you need dense, metric flow vectors.

**Q: Is the fly-brain idea new?**  
A: No. Hassenstein-Reichardt correlators and insect-inspired motion detectors have a long research history. NeuroSight's value is packaging the idea into a small, approachable Python library with connectome-weighted T4/T5-style directions.

**Q: Can I use this on mobile/Raspberry Pi?**  
A: Yes. It's pure Python + NumPy/OpenCV (both available on ARM).

**Q: What if I want higher accuracy?**  
A: Decrease `tile_size` (e.g., to 4 or 8) during initialization for finer spatial resolution, or use the raw motion vectors instead of the saliency map for pixel-level detail.

## License

MIT License. See `LICENSE` for details.

## Citation

If you use NeuroSight in research, please cite:

```bibtex
@software{neurosight,
  title={NeuroSight: Motion Detection Using Fly Optic Lobe Architecture},
  author={CodeForgeNet},
  year={2026},
  url={https://github.com/CodeForgeNet/neurosight}
}
```
