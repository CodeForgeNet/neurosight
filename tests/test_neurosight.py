"""
Phase 5: Unit Tests
Validate motion detection accuracy and edge cases.

Run: pytest test_neurosight.py -v
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path

from neurosight.detector import MotionDetector
from neurosight.utils import FrameBuffer, VideoProcessor, SaliencyMapVisualizer


@pytest.fixture
def temp_connectome():
    """Create a temporary connectome weights file for testing."""
    weights = {
        't4_up': [0.25, 0.25, 0.25, 0.25],
        't4_down': [0.25, 0.25, 0.25, 0.25],
        't4_left': [0.25, 0.25, 0.25, 0.25],
        't4_right': [0.25, 0.25, 0.25, 0.25],
        't5_up': [0.25, 0.25, 0.25, 0.25],
        't5_down': [0.25, 0.25, 0.25, 0.25],
        't5_left': [0.25, 0.25, 0.25, 0.25],
        't5_right': [0.25, 0.25, 0.25, 0.25],
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(weights, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestMotionDetector:
    """Test core motion detection logic."""

    def test_detector_initialization(self, temp_connectome):
        """Test detector initializes with connectome weights."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        assert detector.tile_size == 8
        assert len(detector.connectome_weights) > 0

    def test_on_pathway_response(self, temp_connectome):
        """Test ON pathway (L1) responds to brightness increase."""
        detector = MotionDetector(connectome_weights_path=temp_connectome)
        
        # Dark pixel
        dark = np.array([[0.0, 0.0], [0.0, 0.0]])
        on_dark = detector._rectify_on(dark)
        assert on_dark.max() < 0.2, "ON should be weak for dark pixels"
        
        # Bright pixel
        bright = np.array([[1.0, 1.0], [1.0, 1.0]])
        on_bright = detector._rectify_on(bright)
        assert on_bright.max() > 0.5, "ON should be strong for bright pixels"

    def test_off_pathway_response(self, temp_connectome):
        """Test OFF pathway (L2) responds to brightness decrease."""
        detector = MotionDetector(connectome_weights_path=temp_connectome)
        
        # Dark pixel
        dark = np.array([[0.0, 0.0], [0.0, 0.0]])
        off_dark = detector._rectify_off(dark)
        assert off_dark.max() > 0.5, "OFF should be strong for dark pixels"
        
        # Bright pixel
        bright = np.array([[1.0, 1.0], [1.0, 1.0]])
        off_bright = detector._rectify_off(bright)
        assert off_bright.max() < 0.2, "OFF should be weak for bright pixels"

    def test_no_motion_detection(self, temp_connectome):
        """Test static image (no motion) → near-zero response."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        
        # Two identical frames (static image)
        static_frame = (np.ones((64, 64)) * 128).astype(np.uint8)
        
        saliency, vectors = detector.process_frame(static_frame, static_frame)
        
        # Motion confidence should be near zero
        confidences = [v['confidence'] for v in vectors.values() if v['confidence'] is not None]
        assert len(confidences) == 0 or np.mean(confidences) < 0.1, "Static image should have no motion"

    def test_no_false_positive_on_noise(self, temp_connectome):
        """Test random noise does not produce high-confidence motion."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        
        # Two completely random noise frames
        np.random.seed(42)
        frame1 = (np.random.rand(64, 64) * 255).astype(np.uint8)
        frame2 = (np.random.rand(64, 64) * 255).astype(np.uint8)
        
        saliency, vectors = detector.process_frame(frame1, frame2)
        
        # Even if some random patch triggers a direction, confidence must be low due to inconsistency
        confidences = [v['confidence'] for v in vectors.values() if v['confidence'] is not None]
        if confidences:
            assert np.mean(confidences) < 0.2, "Noise should not produce high confidence motion"

    def test_rightward_motion_detection(self, temp_connectome):
        """Test rightward moving object is detected correctly."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        
        # Create two frames with rightward motion
        frame1 = np.zeros((64, 64), dtype=np.uint8)
        frame1[25:35, 20:30] = 255  # Bright square on left
        
        frame2 = np.zeros((64, 64), dtype=np.uint8)
        frame2[25:35, 22:32] = 255  # Bright square moved right by 2 pixels
        
        saliency, vectors = detector.process_frame(frame1, frame2)
        
        # Check that 'right' direction is detected in some tiles
        directions = [v['direction'] for v in vectors.values() if v['direction'] is not None]
        assert 'right' in directions, "Should detect rightward motion"
        
        # Saliency map should have high values where motion occurs
        assert saliency.max() > 0.1, "Saliency should show motion region"

    def test_saliency_map_shape(self, temp_connectome):
        """Test saliency map has correct shape."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        
        frame1 = (np.random.rand(128, 128) * 255).astype(np.uint8)
        frame2 = (np.random.rand(128, 128) * 255).astype(np.uint8)
        
        saliency, _ = detector.process_frame(frame1, frame2)
        
        assert saliency.shape == (128, 128), "Saliency map shape should match frame"
        assert saliency.min() >= 0 and saliency.max() <= 1, "Saliency should be normalized [0, 1]"


class TestFrameBuffer:
    """Test frame buffering logic."""

    def test_buffer_initialization(self):
        """Test buffer initializes empty."""
        buffer = FrameBuffer(buffer_size=2)
        assert not buffer.is_ready()

    def test_buffer_push_and_full(self):
        """Test pushing frames fills buffer."""
        buffer = FrameBuffer(buffer_size=2)
        
        frame1 = np.zeros((64, 64))
        is_ready = buffer.push(frame1)
        assert not is_ready, "Buffer not full after 1 frame"
        
        frame2 = np.ones((64, 64))
        is_ready = buffer.push(frame2)
        assert is_ready, "Buffer should be full after 2 frames"

    def test_buffer_get_pair(self):
        """Test retrieving frame pair."""
        buffer = FrameBuffer(buffer_size=2)
        
        frame1 = np.zeros((64, 64))
        frame2 = np.ones((64, 64))
        
        buffer.push(frame1)
        buffer.push(frame2)
        
        prev, curr = buffer.get_pair()
        
        assert np.allclose(prev, frame1), "Previous frame should be first"
        assert np.allclose(curr, frame2), "Current frame should be second"

    def test_buffer_circular(self):
        """Test buffer maintains circular FIFO."""
        buffer = FrameBuffer(buffer_size=2)
        
        f1 = np.ones((4, 4)) * 1
        f2 = np.ones((4, 4)) * 2
        f3 = np.ones((4, 4)) * 3
        
        buffer.push(f1)
        buffer.push(f2)
        buffer.push(f3)  # f1 should be dropped
        
        prev, curr = buffer.get_pair()
        
        assert np.allclose(prev, f2), "Buffer should have dropped oldest frame"
        assert np.allclose(curr, f3), "Current should be newest"


class TestVideoProcessor:
    """Test video I/O utilities."""

    def test_grayscale_conversion(self):
        """Test RGB to grayscale conversion."""
        processor = VideoProcessor()
        
        # Create RGB image (red)
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Red channel
        
        gray = processor.to_grayscale(rgb)
        
        assert gray.ndim == 2, "Grayscale should be 2D"
        assert gray.shape == (10, 10), "Shape should match"

    def test_intensity_normalization(self):
        """Test [0, 255] → [0, 1] normalization."""
        processor = VideoProcessor()
        
        frame_8bit = np.array([[0, 128, 255]], dtype=np.uint8)
        frame_normalized = processor.normalize_intensity(frame_8bit)
        
        assert frame_normalized.min() == 0.0
        assert frame_normalized.max() == 1.0
        assert np.isclose(frame_normalized[0, 1], 128/255)


class TestSaliencyVisualization:
    """Test visualization utilities."""

    def test_colormap_application(self):
        """Test applying colormap to saliency."""
        saliency = np.random.rand(64, 64)  # [0, 1]
        
        colored = SaliencyMapVisualizer.apply_colormap(saliency)
        
        assert colored.ndim == 3, "Output should be RGB"
        assert colored.shape[2] == 3, "Should have 3 channels"
        assert colored.shape[0] == 64, "Height should match"

    def test_vector_overlay(self):
        """Test drawing motion vectors on frame."""
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
        
        motion_vectors = {
            '(0, 0)': {
                'direction': 'right',
                'confidence': 0.8,
                'y': 0,
                'x': 0
            },
            '(1, 1)': {
                'direction': 'down',
                'confidence': 0.6,
                'y': 8,
                'x': 8
            }
        }
        
        output = SaliencyMapVisualizer.overlay_vectors(frame, motion_vectors, tile_size=8)
        
        assert output.shape == frame.shape, "Output shape should match input"
        assert not np.array_equal(output, frame), "Output should be modified (arrows drawn)"


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_end_to_end_motion_detection(self, temp_connectome):
        """Test full motion detection pipeline on synthetic video."""
        from neurosight.neurosight_api import NeuroSightDetector
        
        detector = NeuroSightDetector(connectome_path=temp_connectome, tile_size=16)
        
        # Create synthetic frames with clear motion
        frames = []
        for t in range(5):
            frame = np.zeros((128, 128), dtype=np.uint8)
            # Moving bright square
            x = 30 + t * 10
            frame[50:70, x:x+20] = 200
            frames.append(frame)
        
        # Process frame pairs
        for i in range(len(frames) - 1):
            saliency, vectors = detector.process_frame_pair(frames[i], frames[i+1])
            
            assert saliency.shape == (128, 128), "Saliency shape should match frames"
            assert len(vectors) > 0, "Should detect motion vectors"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
