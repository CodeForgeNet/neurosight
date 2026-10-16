"""
NeuroSight: Motion detection using fly optic lobe architecture.

A lightweight, CPU-friendly motion detection library based on the Virtual Fly Brain
connectome. Implements the Hassenstein-Reichardt correlator using real synapse weights
from the fly T4/T5 direction-selective neurons.

Example:
    >>> from neurosight import NeuroSightDetector
    >>> detector = NeuroSightDetector(connectome_path="connectome_weights.json")
    >>> saliency, vectors = detector.process_video("input.mp4")
"""

__version__ = "0.1.1"
__author__ = "CodeForgeNet"

from .detector import MotionDetector, MotionResult, NeuroSightConfig
from .neurosight_api import NeuroSightDetector
from .utils import FrameBuffer, SaliencyMapVisualizer, VideoProcessor

__all__ = [
    'NeuroSightDetector',
    'MotionDetector',
    'MotionResult',
    'NeuroSightConfig',
    'FrameBuffer',
    'VideoProcessor',
    'SaliencyMapVisualizer',
]

logger_setup = False

def setup_logging(level: str = "INFO"):
    """Configure logging for the library."""
    import logging
    global logger_setup
    
    if not logger_setup:
        logging.basicConfig(
            level=getattr(logging, level),
            format='[%(name)s] %(levelname)s: %(message)s'
        )
        logger_setup = True
