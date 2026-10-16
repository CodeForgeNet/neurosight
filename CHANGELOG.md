# Changelog

All notable changes to NeuroSight are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
