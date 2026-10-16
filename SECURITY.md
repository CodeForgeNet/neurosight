# Security Policy

## Supported versions

Security fixes are applied to the latest released minor version.

## Reporting a vulnerability

Please report suspected vulnerabilities privately using GitHub's
[security advisories](https://github.com/CodeForgeNet/neurosight/security/advisories/new)
rather than a public issue.

Include, where possible:

- A description of the issue and its impact.
- Steps to reproduce (a minimal example is ideal).
- Affected versions and environment (OS, Python, NumPy/OpenCV).

We aim to acknowledge reports within a few business days.

## Scope notes

NeuroSight is a local, CPU-only library that performs no network calls at
runtime. The most likely issue classes are:

- Parsing of user-supplied connectome weight files (JSON).
- Handling of untrusted video/frame inputs passed to OpenCV.
- Resource exhaustion (memory/CPU) from very large frames or tile configurations.

Callers are responsible for validating and trusting the paths, videos, and frame
arrays they supply.
