# Changelog

All notable changes to NeuroSight are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-23

### Breaking changes
- Robotics outputs (`regions`, `looming_score`, `time_to_contact`, `steering_signal`) are disabled by default. Enable them explicitly or use `preset_looming()`. Values are `[]`, `0.0`, `None`, and a zeroed steering dict unless enabled.
- `NeuroSightDetector.process_video()` no longer returns per-frame tile vectors by default. Pass `return_vectors=True` to retain them; the returned dict is empty otherwise. Previously memory grew with video length.
- `process_pair()` now uses local spatial high-pass features plus a temporal-change gate instead of the old signal path. Its magnitudes, confidences, and direction maps differ from 0.1.x, and it is independent of streaming state.
- The default `scales` changed from `(1, 2)` to `(1, 2, 4, 8)`. Multi-scale fusion now selects the smallest scale that explains each tile instead of averaging all scales, and the pair-path high-pass width now follows the tile size. Outputs and per-frame cost differ from 0.1.x.
- An explicitly supplied but missing `connectome_weights_path` now raises `FileNotFoundError` instead of silently switching to bundled weights.
- Custom connectome weight files are validated strictly: they must define all four T4/T5 directions (subtypes `a`–`d` or `up`/`down`/`left`/`right`) and every vector must have the same length. Files that previously loaded with missing directions or mixed lengths now raise `ValueError` instead of falling back to uniform gains.
- Removed the `gui` extra because it installed `opencv-python` alongside the core `opencv-python-headless` dependency; only one OpenCV distribution should be installed per environment.

### Added
- Speed/phase sweep in `benchmarks/run_optical_flow_benchmark.py`, covering 1–16 px/frame and four sub-tile phases for both synthetic stimuli.
- Regression tests for pair-path speed range, response stability, and broad-grating preservation.
- `process_video()` documents and returns the per-pixel maximum saliency over all frames, not the last frame.

### Changed
- The stateless pair path now resolves direction from roughly 1–14 px/frame on the synthetic speed sweep (previously documented as roughly 1–3 px/frame). The stateful path remains tuned for slower motion.
- The stateful detector now correlates temporal contrast (previous vs. current frame) instead of raw intensity. Static structure no longer contributes motion energy, and the delay line is seeded with real history rather than the current frame. As a consequence, after `reset()` the stateful path needs two frames before it can report motion: the first seeds the frame history, the second establishes temporal contrast, and the third produces the first nonzero correlation. The stateless pair path is unaffected.
- Lateral-inhibition tile means are smoothly upsampled, removing tile-aligned seams that could create false perpendicular motion.
- `MotionResult.confidence_map` aliases `saliency_map`, and `ego_compensated_field` aliases `motion_field` when ego-motion compensation is off. Treat result arrays as read-only; mutating one alias changes the other field.
- Connectome weights are validated structurally (all four T4/T5 directions, consistent vector lengths, finite values).
- `preset_edge_fast()` now explicitly disables regions and steering, matching its documentation.
- Benchmark direction summaries are confidence-weighted instead of raw tile counts, so many weak noisy tiles cannot outvote a few strong ones.

### Fixed
- `process_video()` resets detector state at the start of each call, so the first frame of a second video no longer compares against the last frame of the first.
- `process_pair()` no longer depends on streaming state. It uses fixed default rectifier parameters, so the same frames always produce the same output.
- Frames containing NaN or Inf are rejected with `ValueError` before they can corrupt the adaptive threshold.
- Correlation masking now uses a frame-border validity mask instead of tile-local masks, removing blind spots at displacements that align with the tile grid.
- Broad, slowly varying patterns survive the pair-path high-pass instead of vanishing before correlation.
- README webcam example and `examples/steering_and_looming.py` set `input_color="bgr"`.
- `steering_and_looming.py` reports TTC in seconds and no longer crashes when TTC is unavailable.
- CI now runs the test matrix on Linux, macOS, and Windows.

## [0.1.2] - 2026-10-17

### Fixed
- `process_video` no longer returns `None` for very short or single-frame videos.
- Output video now inherits the source container frame rate instead of a hardcoded 30 FPS.
- Streaming timestamps now use the real frame interval, keeping the temporal low-pass calibrated.
- `MotionDetector.reset()` restores the adaptive sigmoid threshold.
- Connectome weight loading validates payloads and raises clear errors for malformed files.
- Contrast normalization no longer explodes on near-black frames; pair-path flicker suppression is now symmetric.
- `time_to_contact` is expressed in seconds when a frame interval is known.

### Changed
- Direction/T4/T5 maps are expanded lazily, reducing per-frame memory use.
- `connectome_weights_path`/`connectome_path` default to `None` (bundled weights) instead of a relative file path.
- Cache the per-shape shift masks with a bounded LRU-style cache.
- Dropped the unused `scipy` runtime dependency; core now depends on `opencv-python-headless`.
- Version is single-sourced from installed package metadata.

### Added
- Conformance/release CI: Python 3.10–3.13 matrix, ruff, mypy, coverage gate, build + `twine check` smoke test.
- PEP 561 `py.typed` marker.
- Config validation with clear error messages.
- Regression tests for the video pipeline, weight loading, lazy maps, and TTC units.

## [0.1.1]

- Documentation, packaging, and naming polish.

## [0.1.0]

- Initial public release: fly-inspired motion detection, connectome-weighted T4/T5 directions, robotics outputs.
