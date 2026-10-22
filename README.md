# NeuroSight

[![PyPI version](https://badge.fury.io/py/neurosight.svg)](https://badge.fury.io/py/neurosight)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/CodeForgeNet/neurosight/actions/workflows/build.yml/badge.svg)](https://github.com/CodeForgeNet/neurosight/actions)

**Lightweight motion detection inspired by the fly optic lobe.**

![NeuroSight Demo](https://raw.githubusercontent.com/CodeForgeNet/neurosight/main/assets/demo.gif)

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

The core install uses `opencv-python-headless`. The interactive examples use
`cv2.imshow`/webcam capture and therefore need a GUI-enabled OpenCV build. In a
dedicated virtual environment, replace the headless distribution before running
those examples:

```bash
pip uninstall -y opencv-python-headless
pip install "opencv-python>=4.8"
```

Do not install `opencv-python` and `opencv-python-headless` side by side; both
provide the same `cv2` module and the result depends on install order.

## How to Use It

NeuroSight ships with pre-extracted biological connectome weights built directly into the library, so you can start detecting motion immediately.

### Quick Example: Process a Video

The simplest way to use NeuroSight is to process a video file and output an annotated video with a motion heatmap and directional arrows.

```python
from neurosight import NeuroSightDetector

detector = NeuroSightDetector()

# `aggregate_saliency` is the per-pixel maximum over all frames, useful for a
# summary heatmap. Pass `return_vectors=True` if you also need the per-frame
# tile dictionaries (they grow linearly with video length).
aggregate_saliency, _ = detector.process_video(
    "input.mp4",
    output_video="output_with_motion.mp4",
    overlay_vectors=True
)
```

### Real-time / Frame-by-Frame Processing

If you have exactly two frames, use the stateless pair API:

```python
from neurosight.detector import MotionDetector

detector = MotionDetector()
result = detector.process_pair(previous_frame, current_frame)

print(result.global_vector)
saliency = result.saliency_map
```

If you are reading from a webcam or integrating into a larger robotics pipeline, use the stateful API.

```python
import cv2
import time
from neurosight.detector import MotionDetector, NeuroSightConfig

# Use a preset for robust edge-device processing. Looming is opt-in because
# it is a research heuristic, not a validated safety signal.
config = NeuroSightConfig.preset_robust(
    tile_size=8,
    input_color="bgr",  # OpenCV delivers BGR frames
    enable_looming=True,
)
detector = MotionDetector(config=config)
cap = cv2.VideoCapture(0)

while True:
    ret, frame_curr = cap.read()
    if not ret: break
    
    # Update the stateful detector
    result = detector.update(frame_curr, timestamp=time.time())
    
    if result is not None:
        # Display the result
        cv2.imshow("Motion Saliency", (result.saliency_map * 255).astype('uint8'))
        
        # Check robotics outputs (looming must be enabled in the config)
        if result.looming_score and result.looming_score > 0.05:
            if result.time_to_contact is not None:
                print(f"Expansion detected. Heuristic TTC: {result.time_to_contact:.2f}s")
            else:
                print("Expansion detected; time-to-contact unavailable.")
            
    if cv2.waitKey(1) == ord('q'): break
```

### Parameter Presets

NeuroSight ships with several config presets to help you tune for your use case immediately:

| Preset Method | Use Case | Features |
|--------------|----------|----------|
| `preset_education()` | Visualizing basic biological principles. | Slow but highly explainable. Robotics/robustness features disabled. |
| `preset_edge_fast()` | High-FPS operation on Raspberry Pi/Edge. | Single spatial scale. Heavy denoising and regions disabled. |
| `preset_robust()` | Handheld cameras, noisy environments. | Heavy temporal denoising, ego-motion, and flicker suppression on. |
| `preset_looming()` | Expansion/TTC experiments (research, not safety-rated). | Multi-scale enabled. Region extraction, looming, and steering enabled. |

You can find more detailed runnable examples in the `examples/` directory!

## What it Provides

The core detector returns a structured `MotionResult` dataclass containing:

### 1. Motion Fields & Saliency
- `saliency_map`: A 2D NumPy array representing the intensity of motion detected (scaled from `0.0` to `1.0`).
- `motion_field`: A `(H, W, 2)` NumPy array representing the horizontal and vertical unitless motion energy.
- `direction_maps`: A dictionary of `up`, `down`, `left`, `right` response grids.

### 2. Robotics & Navigation (Optional)
These outputs are disabled by default. Enable them explicitly or use `preset_looming()`.
- `regions`: A list of dictionaries representing isolated moving components (includes `bbox`, `centroid`, `area`, `dominant_vector`).
- `looming_score`: Dimensionless expansion evidence score (research heuristic).
- `time_to_contact`: Heuristic time-to-contact estimate; not a calibrated or safety-rated signal.
- `steering_signal`: A dictionary containing a `-1.0` to `1.0` `turn` signal (avoidance hook), `forward_caution`, and `collision_urgency`.

### 3. Legacy Dictionary
- `motion_vectors`: Lazy-evaluated backward compatibility property that computes per-tile vector summaries (slower).

## Architecture & Biological Background

Fly-inspired motion detection is an established research area. NeuroSight packages one practical version of that idea as a small Python library, combining a Hassenstein-Reichardt-style correlator with pre-extracted T4/T5 weighting:

- **T4/T5 Neurons**: The core of fly motion vision. T4 neurons detect moving bright edges (ON pathway), and T5 neurons detect moving dark edges (OFF pathway). Each is divided into 4 subtypes corresponding to the 4 cardinal directions (up, down, left, right).
- **Medulla Interneurons**: Signals from photoreceptors are split into L1 (light-sensitive) and L2 (dark-sensitive) channels. They pass through medulla interneurons (Mi1, Tm3, etc.) which act as temporal delay lines.
- **Hassenstein-Reichardt Correlator**: By multiplying a delayed signal from one photoreceptor with an undelayed signal from a neighboring one, the brain determines direction and speed.
- **Connectome Weights**: The included weights are derived from *Drosophila* hemibrain connectivity and normalized for this implementation.

### Data provenance

The bundled `t4_*`/`t5_*` weighting vectors are normalized summaries of
*Drosophila* hemibrain (Janelia/NeuPrint) connectivity, computed offline and
shipped as static data (`neurosight/data/connectome_weights.json`, mirrored in
`neurosight/connectome_weights_data.py`). The library performs **no network
calls at runtime**.

The upstream extraction script is not included in this repository or the
released package. The redistribution terms for these derived weights have not
been independently reviewed here. If you plan to redistribute or use them
commercially, verify the current Janelia/NeuPrint data terms and, if necessary,
regenerate the weights from a source you are permitted to use.

## Scientific Scope and Limitations

NeuroSight is an engineering approximation inspired by insect motion vision,
not a biophysical simulation or a certified sensor:

- The correlator is a simplified Hassenstein-Reichardt-style model. The bundled
  weights scale the ON/OFF pathway contributions; they do not reproduce the
  full medulla/lobula plate circuit.
- `motion_field` values are unitless motion energy, not calibrated pixel
  displacement. Do not interpret them as metric optical flow.
- `looming_score` and `time_to_contact` are research heuristics. They are
  disabled by default, and `time_to_contact` is **not** a validated collision
  warning or safety signal.
- Directional benchmark results apply to the included synthetic stimuli. They
  are not evidence of general real-world accuracy.
- With the default `scales=(1, 2, 4, 8)`, the stateless `process_pair()` path
  recovered direction at every sampled sub-tile phase from 1 to 14 px/frame on
  the synthetic translating square and the 32 px-period grating. The stateful
  `update()` path is limited by its temporal delay line and was reliable to
  about 4 px/frame for small isolated objects in the same sweep. It also needs
  two frames after `reset()` before it can report motion (the first seeds frame
  history; the second establishes temporal contrast). Faster motion needs
  downscaled input, a lower frame rate, or additional scales. A 32 px-period
  grating is inherently ambiguous at 16 px/frame (half a period).

## Performance

Current local smoke check using `NeuroSightConfig.preset_edge_fast(tile_size=8)` on a
640×480 synthetic frame pair (single process, CPU):

| Metric | NeuroSight |
|--------|-----------:|
| Mean latency | 16.70 ms |
| p95 latency | 17.11 ms |
| Mean throughput | 59.9 FPS |

These figures are a local sanity check, not a published hardware benchmark. Real performance depends on frame size, tile size, hardware, OpenCV build, enabled options, and the scene being processed. The table uses the single-scale `preset_edge_fast()`; the default `scales=(1, 2, 4, 8)` configuration costs roughly 60% more CPU per frame in the same local measurement.

### Directional Synthetic Benchmark

NeuroSight also includes a small directional benchmark using both a translating square and a drifting sinusoidal grating across the four cardinal directions:

```bash
python benchmarks/run_optical_flow_benchmark.py
```

The script runs the original eight-case head-to-head comparison at 4 px/frame,
then sweeps speed and sub-tile phase. These results were rerun against the
current stateless pair-path implementation:

| Condition | Result |
|-----------|--------|
| Cardinal directions at 4 px/frame (8 cases) | NeuroSight 100.0%, Farneback mean-flow 50.0% |
| Translating square, 1–16 px/frame (16 cases per speed) | 16/16 correct at every speed |
| Drifting grating (32 px period), 1–14 px/frame | 16/16 correct at every speed |
| Drifting grating at 16 px/frame | 8/16 — half-period ambiguity, not a fixable detector error |

The benchmark sweeps four phase offsets per speed so results do not depend on
where an object falls relative to the 8 px tile grid. This is still a scoped
synthetic benchmark, not a general claim that classical optical flow fails.
Farneback is summarized by a masked mean dense-flow vector, which can collapse
toward zero on periodic gratings with the current parameters.

## FAQ

**Q: Is this better than optical flow?**  
A: Not generally. NeuroSight is complementary: it provides lightweight directional motion saliency and immediate robotics abstractions (steering, looming). Classical and deep optical-flow methods are usually better when you need dense, *calibrated metric* flow vectors (i.e. pixel displacement).

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
