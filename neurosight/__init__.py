"""
NeuroSight: Motion detection using fly optic lobe architecture.

A lightweight, CPU-friendly motion detection library based on the fly
connectome. Implements the Hassenstein-Reichardt correlator using T4/T5
direction-selective weighting derived from connectome data.

Example:
    >>> from neurosight import NeuroSightDetector
    >>> detector = NeuroSightDetector()
    >>> aggregate_saliency, _ = detector.process_video("input.mp4")
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .detector import MotionDetector, MotionResult, NeuroSightConfig
from .neurosight_api import NeuroSightDetector
from .utils import (
    FrameBuffer,
    OutputWriter,
    PerformanceMonitor,
    SaliencyMapVisualizer,
    VideoProcessor,
    VideoProperties,
)

try:  # Version is single-sourced from installed package metadata.
    __version__ = _pkg_version("neurosight")
except PackageNotFoundError:  # Running from a source checkout without installation.
    __version__ = "0.0.0+unknown"

__author__ = "CodeForgeNet"

# Library best practice: never configure logging for the host application.
# Emit nothing unless the caller opts in.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'NeuroSightDetector',
    'MotionDetector',
    'MotionResult',
    'NeuroSightConfig',
    'FrameBuffer',
    'VideoProcessor',
    'VideoProperties',
    'PerformanceMonitor',
    'SaliencyMapVisualizer',
    'OutputWriter',
    '__version__',
]

logger_setup = False


def setup_logging(level: str = "INFO"):
    """Convenience helper to configure logging for interactive use.

    The library never configures logging on import; call this explicitly if you
    want NeuroSight's log messages emitted to the root handler.
    """
    global logger_setup

    if not logger_setup:
        logging.basicConfig(
            level=getattr(logging, level),
            format='[%(name)s] %(levelname)s: %(message)s'
        )
        logger_setup = True
