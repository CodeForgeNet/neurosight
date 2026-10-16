import numpy as np
import pytest

from neurosight.detector import MotionDetector, NeuroSightConfig


def test_extract_regions():
    config = NeuroSightConfig(tile_size=8, enable_regions=True)
    detector = MotionDetector(config=config)
    
    # Create a synthetic saliency map with two distinct active regions
    H, W = 64, 64
    saliency = np.zeros((H, W), dtype=np.float32)
    saliency[10:20, 10:20] = 1.0  # Region 1
    saliency[40:50, 40:50] = 1.0  # Region 2
    
    # Synthetic motion field
    motion = np.zeros((H, W, 2), dtype=np.float32)
    motion[10:20, 10:20, 0] = 1.0 # Right
    motion[40:50, 40:50, 1] = -1.0 # Up
    
    regions = detector._extract_regions(saliency, motion)
    
    assert len(regions) == 2
    
    # Sort regions by y-coordinate to predictably check them
    regions.sort(key=lambda r: r['bbox'][1])
    
    # Region 1 checks
    assert regions[0]['area'] == 100
    assert regions[0]['dominant_vector'][0] == 1.0
    
    # Region 2 checks
    assert regions[1]['area'] == 100
    assert regions[1]['dominant_vector'][1] == -1.0

def test_looming_and_ttc():
    config = NeuroSightConfig(tile_size=8, enable_looming=True)
    detector = MotionDetector(config=config)
    
    H, W = 64, 64
    motion = np.zeros((H, W, 2), dtype=np.float32)
    saliency = np.zeros((H, W), dtype=np.float32)
    
    # Create radial expansion (looming)
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H//2, W//2
    dy = y - cy
    dx = x - cx
    
    # Expansion field
    motion[..., 0] = dx * 0.1
    motion[..., 1] = dy * 0.1
    saliency.fill(1.0)
    
    looming, ttc = detector._compute_looming_and_ttc(motion, saliency)
    
    assert looming > 0.05
    assert ttc is not None
    assert ttc > 0.0

def test_looming_ttc_scales_with_dt():
    """Time-to-contact should be reported in seconds when dt is supplied."""
    config = NeuroSightConfig(tile_size=8, enable_looming=True)
    detector = MotionDetector(config=config)

    H, W = 64, 64
    motion = np.zeros((H, W, 2), dtype=np.float32)
    saliency = np.zeros((H, W), dtype=np.float32)

    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    motion[..., 0] = (x - cx) * 0.1
    motion[..., 1] = (y - cy) * 0.1
    saliency.fill(1.0)

    _, ttc_frames = detector._compute_looming_and_ttc(motion, saliency)
    _, ttc_seconds = detector._compute_looming_and_ttc(motion, saliency, dt=0.033)

    assert ttc_frames is not None and ttc_seconds is not None
    assert ttc_seconds == pytest.approx(ttc_frames * 0.033)


def test_steering_signal():
    config = NeuroSightConfig(tile_size=8, enable_steering=True)
    detector = MotionDetector(config=config)
    
    H, W = 64, 64
    saliency = np.zeros((H, W), dtype=np.float32)
    
    # Obstacle on the left -> turn right (positive)
    saliency[:, :32] = 1.0
    
    steering = detector._compute_steering_signal(saliency)
    
    assert steering['turn'] == 1.0
    assert steering['forward_caution'] > 0.5
    assert steering['collision_urgency'] > 0.5
    
    # Obstacle on the right -> turn left (negative)
    saliency.fill(0)
    saliency[:, 32:] = 1.0
    
    steering2 = detector._compute_steering_signal(saliency)
    assert steering2['turn'] == -1.0
