import pytest
import numpy as np
import cv2
from neurosight.detector import MotionDetector, NeuroSightConfig

@pytest.fixture
def temp_connectome(tmp_path):
    import json
    weights = {
        't4_a': [0.25]*4, 't4_b': [0.25]*4, 't4_c': [0.25]*4, 't4_d': [0.25]*4,
        't5_a': [0.25]*4, 't5_b': [0.25]*4, 't5_c': [0.25]*4, 't5_d': [0.25]*4,
    }
    path = tmp_path / "connectome.json"
    with open(path, "w") as f:
        json.dump({'weights': weights}, f)
    return str(path)

def generate_square_scene(H, W, square_pos, bg_val=100, fg_val=200):
    frame = np.full((H, W), bg_val, dtype=np.float32)
    y, x = square_pos
    frame[y:y+16, x:x+16] = fg_val
    return frame / 255.0

def test_ego_motion_compensation(temp_connectome):
    """Test that global camera shake is subtracted from local object motion."""
    config = NeuroSightConfig(tile_size=8, ego_motion_compensation=True, motion_threshold=0.005, adaptation_rate=1.0)
    detector = MotionDetector(connectome_weights_path=temp_connectome, config=config)
    
    # Simulate a scene where the camera pans Right (so everything moves Left)
    # AND an object in the scene moves Right (relative to camera, it stays still or moves Right)
    # Actually, let's just make the whole background move Left, and one object moves Right.
    # Camera moving Right means background moves Left (-1, 0).
    # Object moves Right (+1, 0).
    
    frame1 = np.zeros((64, 64), dtype=np.float32)
    # Textured background
    frame1[10:20, 10:20] = 0.5
    frame1[40:50, 40:50] = 0.6
    frame1[10:20, 40:50] = 0.7
    # Moving object
    frame1[30:40, 30:40] = 0.9
    
    # Frame 2: Background shifts Left (dx=-2), Object shifts Right (dx=+2)
    frame2 = np.zeros((64, 64), dtype=np.float32)
    frame2[10:20, 8:18] = 0.5
    frame2[40:50, 38:48] = 0.6
    frame2[10:20, 38:48] = 0.7
    frame2[30:40, 32:42] = 0.9
    
    frame3 = np.zeros((64, 64), dtype=np.float32)
    frame3[10:20, 6:16] = 0.5
    frame3[40:50, 36:46] = 0.6
    frame3[10:20, 36:46] = 0.7
    frame3[30:40, 34:44] = 0.9

    frame4 = np.zeros((64, 64), dtype=np.float32)
    frame4[10:20, 4:14] = 0.5
    frame4[40:50, 34:44] = 0.6
    frame4[10:20, 34:44] = 0.7
    frame4[30:40, 36:46] = 0.9

    # Warm up
    detector.update(frame1)
    detector.update(frame2)
    detector.update(frame3)
    res = detector.update(frame4)

    assert res is not None
    # Global vector should be roughly left (-x)
    assert res.ego_motion_vector[0] < -0.01, f"Global vector should be Left (negative X), got {res.ego_motion_vector[0]}"
    
    # In the raw motion field, the object (at y=30:40, x=30:40) moves Right (+x)
    # In the ego compensated field, the background (moving Left) should be near zero!
    bg_y, bg_x = 1, 0
    raw_bg_vx = res.motion_field[bg_y*8, bg_x*8, 0]
    comp_bg_vx = res.ego_compensated_field[bg_y*8, bg_x*8, 0]

    assert raw_bg_vx < -0.01, "Raw background motion should be Left"
    assert abs(comp_bg_vx) < abs(raw_bg_vx), "Ego compensation should reduce background motion"

def test_flicker_suppression(temp_connectome):
    """Test that a global illumination flash is suppressed."""
    config = NeuroSightConfig(tile_size=8, flicker_suppression=True)
    detector = MotionDetector(connectome_weights_path=temp_connectome, config=config)
    
    frame1 = np.full((32, 32), 0.2, dtype=np.float32)
    # Add a small edge
    frame1[10:20, 10:20] = 0.8
    
    frame2 = np.full((32, 32), 0.5, dtype=np.float32) # massive global flash
    frame2[10:20, 10:20] = 0.8 # edge stays same
    
    detector.update(frame1)
    res = detector.update(frame2)
    
    # Without suppression, this flash might trigger massive ON-pathway activation
    # With suppression, max magnitude should remain low
    assert res.magnitude_map.max() < 0.2, "Flicker should be suppressed"

def test_exposure_penalty(temp_connectome):
    """Test that overexposed scenes reduce confidence."""
    config = NeuroSightConfig(tile_size=8)
    detector = MotionDetector(connectome_weights_path=temp_connectome, config=config)
    
    frame1 = np.full((32, 32), 0.99, dtype=np.float32) # Overexposed
    frame2 = np.full((32, 32), 0.99, dtype=np.float32)
    frame2[10:20, 15:25] = 0.9 # slight movement but still mostly overexposed
    
    detector.update(frame1)
    res = detector.update(frame2)
    
    assert res.scene_quality['exposure_clip'] > 0.8, "Scene should be flagged as overexposed"
    assert res.confidence_map.max() < 0.2, "Confidence should be penalized for overexposure"

def test_blur_handling(temp_connectome):
    """Test that blur handling smooths out high frequency noise."""
    config = NeuroSightConfig(tile_size=8, blur_handling=True)
    detector = MotionDetector(connectome_weights_path=temp_connectome, config=config)
    
    np.random.seed(42)
    frame1 = np.random.rand(32, 32).astype(np.float32)
    frame2 = np.random.rand(32, 32).astype(np.float32)
    
    detector.update(frame1)
    res = detector.update(frame2)
    
    assert res is not None
    # Just asserting it runs without error and returns low confidence for random noise
    assert res.confidence_map.max() < 0.5
