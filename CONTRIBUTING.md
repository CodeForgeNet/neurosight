# Contributing to NeuroSight

Thanks for your interest in improving NeuroSight!

## Development setup

```bash
git clone https://github.com/CodeForgeNet/neurosight
cd neurosight
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Interactive examples (webcam / `cv2.imshow`) additionally need the GUI build of
OpenCV: `pip install -e ".[dev,gui]"`.

## Before opening a pull request

Run the same checks CI runs:

```bash
ruff check neurosight tests benchmarks
mypy neurosight
pytest --cov=neurosight --cov-report=term-missing --cov-fail-under=80
```

## Guidelines

- Keep changes focused; one logical change per pull request.
- Add or update tests for behavior changes. New code should not lower coverage below the 80% gate.
- Keep the public API (`neurosight.__all__`) stable; document deprecations.
- The detector core is stateful and CPU-oriented — prefer vectorized NumPy/OpenCV over Python loops.
- Follow the existing style (type hints, docstrings, no logging configuration at import time).

## Reporting issues

Please include the NeuroSight version, Python/NumPy/OpenCV versions, platform,
and a minimal reproduction (ideally a synthetic frame pair).

## Code of Conduct

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).
