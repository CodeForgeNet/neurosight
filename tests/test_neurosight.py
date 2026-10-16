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

from neurosight.detector import MotionDetector, MotionResult
from neurosight.utils import FrameBuffer, VideoProcessor, SaliencyMapVisualizer


def make_translating_square(direction, size=64, speed=2):
    """Create a simple two-frame cardinal translation stimulus."""
    dx, dy = 0, 0
    if direction == 'right':
        dx = speed
    elif direction == 'left':
        dx = -speed
    elif direction == 'down':
        dy = speed
    elif direction == 'up':
        dy = -speed
    else:
        raise ValueError(f"unsupported direction: {direction}")

    frame1 = np.zeros((size, size), dtype=np.uint8)
    frame2 = np.zeros((size, size), dtype=np.uint8)
    y0, y1 = 24, 36
    x0, x1 = 24, 36
    frame1[y0:y1, x0:x1] = 255
    frame2[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = 255
    return frame1, frame2


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
        """Test OFF pathway responds to contrast decrements (passed as positive contrast)."""
        detector = MotionDetector(connectome_weights_path=temp_connectome)
    
        # A 0.5 decrease in brightness, which is passed in as positive contrast 0.5
        contrast_decrement = np.array([[0.5, 0.5], [0.5, 0.5]])
        off_response = detector._rectify_off(contrast_decrement)
        assert off_response.max() > 0.5, "OFF should be strong for contrast decrements"
        
        # Bright pixel (positive contrast is clipped to 0 for OFF pathway)
        bright = np.array([[0.0, 0.0], [0.0, 0.0]])
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
        frame1[25:35, 20:30] = 255
        frame2 = np.zeros((64, 64), dtype=np.uint8)
        frame2[25:35, 22:32] = 255
        frame3 = np.zeros((64, 64), dtype=np.uint8)
        frame3[25:35, 24:34] = 255
        frame4 = np.zeros((64, 64), dtype=np.uint8)
        frame4[25:35, 26:36] = 255
        
        detector.update(frame1)
        detector.update(frame2)
        detector.update(frame3)
        res = detector.update(frame4)
        saliency, vectors = res.as_legacy()
        
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

    def test_process_pair_returns_structured_result(self, temp_connectome):
        """Test the forward-compatible API returns motion field and diagnostics."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)

        frame1 = np.zeros((32, 32), dtype=np.uint8)
        frame2 = np.zeros((32, 32), dtype=np.uint8)
        frame1[12:20, 8:16] = 255
        frame2[12:20, 10:18] = 255
        result = detector.process_pair(frame1, frame2)

        assert isinstance(result, MotionResult)
        assert result.saliency_map.shape == (32, 32)
        assert result.confidence_map.shape == (32, 32)
        assert result.magnitude_map.shape == (32, 32)
        assert result.angle_map.shape == (32, 32)
        assert result.motion_field.shape == (32, 32, 2)
        assert set(result.direction_maps) == {'up', 'down', 'left', 'right'}
        assert result.diagnostics['total_tiles'] == 16
        assert result.global_vector[0] > 0

    @pytest.mark.parametrize("direction", ["right", "left", "up", "down"])
    def test_process_pair_cardinal_directions(self, temp_connectome, direction):
        """Test direct two-frame API detects all cardinal directions."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        frame1, frame2 = make_translating_square(direction)

        result = detector.process_pair(frame1, frame2)
        directions = [
            v['direction']
            for v in result.motion_vectors.values()
            if v['direction'] is not None and v['confidence'] > 0.01
        ]

        assert direction in directions
        assert result.diagnostics['active_tiles'] > 0

    @staticmethod
    def _dominant_direction(vectors):
        from collections import Counter
        valid = [v['direction'] for v in vectors.values() if v['direction']]
        return Counter(valid).most_common(1)[0][0] if valid else 'none'

    @pytest.mark.parametrize("direction", ["right", "left", "up", "down"])
    def test_streaming_and_pair_modes_agree_on_direction(self, direction):
        """Contract test: stateless pair and adaptive streaming modes agree on direction.

        The two code paths use different signal domains (raw intensity vs filtered
        contrast) so magnitudes differ, but their directional verdict must match.
        """
        detector = MotionDetector(tile_size=8)
        frames = []
        for t in range(5):
            frame = np.zeros((64, 64), dtype=np.uint8)
            dx = {"right": 1, "left": -1}.get(direction, 0) * t * 2
            dy = {"down": 1, "up": -1}.get(direction, 0) * t * 2
            frame[24 + dy:36 + dy, 24 + dx:36 + dx] = 255
            frames.append(frame)

        pair_dir = self._dominant_direction(
            detector.process_pair(frames[-2], frames[-1]).motion_vectors
        )

        detector.reset()
        for frame in frames[:-1]:
            detector.update(frame)
        stream_result = detector.update(frames[-1])
        stream_dir = self._dominant_direction(stream_result.motion_vectors)

        assert pair_dir == direction
        assert stream_dir == direction

    def test_process_pair_static_active_tiles_below_one_percent(self, temp_connectome):
        """Test static scenes do not produce meaningful active tiles."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        frame = np.full((64, 64), 128, dtype=np.uint8)

        result = detector.process_pair(frame, frame)
        active_ratio = result.diagnostics['active_tiles'] / result.diagnostics['total_tiles']

        assert active_ratio < 0.01
        assert np.max(result.magnitude_map) == 0.0

    def test_process_pair_uint8_float_consistency(self, temp_connectome):
        """Test uint8 and normalized-float inputs produce the same dominant direction."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        frame1, frame2 = make_translating_square('right')

        uint8_result = detector.process_pair(frame1, frame2)
        float_result = detector.process_pair(
            frame1.astype(np.float32) / 255.0,
            frame2.astype(np.float32) / 255.0,
        )

        assert uint8_result.global_vector[0] > 0
        assert float_result.global_vector[0] > 0
        assert np.sign(uint8_result.global_vector[0]) == np.sign(float_result.global_vector[0])

    def test_benchmark_core_cases_remain_correct(self):
        """Regression test for the benchmark cases that previously fell to 0%."""
        from benchmarks.run_optical_flow_benchmark import (
            generate_synthetic_motion,
            generate_translating_square,
            summarize_neurosight_vectors,
        )

        detector = MotionDetector(
            connectome_weights_path='neurosight/data/connectome_weights.json',
            tile_size=8,
        )
        generators = [generate_translating_square, generate_synthetic_motion]
        directions = ['right', 'left', 'up', 'down']

        for generator in generators:
            for direction in directions:
                frame1, frame2, _, _ = generator(width=64, height=64, speed=2, direction=direction)
                result = detector.process_pair(frame1, frame2)
                predicted, _, _ = summarize_neurosight_vectors(result.motion_vectors)
                assert predicted == direction

    def test_invalid_frame_pair_shape_raises(self, temp_connectome):
        """Test mismatched input frames fail with a clear error."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)

        frame1 = np.zeros((32, 32), dtype=np.uint8)
        frame2 = np.zeros((32, 31), dtype=np.uint8)

        with pytest.raises(ValueError, match="frame shapes must match"):
            detector.process_pair(frame1, frame2)

    def test_shift_no_wrap_zero_pads_edges(self):
        """Test spatial shifts do not create artificial wraparound pixels."""
        frame = np.zeros((3, 3), dtype=np.float32)
        frame[0, 0] = 1.0

        shifted = MotionDetector._shift_no_wrap(frame, 0, 1)

        assert shifted[0, 1] == 1.0
        assert shifted[0, 0] == 0.0
        assert shifted[0, 2] == 0.0


    def test_bright_edge_vs_dark_edge(self, temp_connectome):
        """Phase 4: Bright-edge and dark-edge tests selectively activate intended pathways."""
        from neurosight.detector import NeuroSightConfig
        config = NeuroSightConfig(tile_size=8, lateral_inhibition=False) # Disable lateral inhibition to test pure edge response
        detector = MotionDetector(connectome_weights_path=temp_connectome, config=config)
        
        # Frame 1: Gray background
        frame1 = np.full((32, 32), 128, dtype=np.uint8)
        
        # Frame 2: Bright square appears
        frame2_bright = frame1.copy()
        frame2_bright[12:20, 12:20] = 255
        
        # Frame 2: Dark square appears
        frame2_dark = frame1.copy()
        frame2_dark[12:20, 12:20] = 0
        
        res_bright = detector.process_pair(frame1, frame2_bright)
        
        # Get raw t4 and t5 maps
        # Since it's a symmetric expansion from center, we just check overall magnitude
        t4_bright = np.sum([v for v in res_bright.t4_maps.values()])
        t5_bright = np.sum([v for v in res_bright.t5_maps.values()])
        
        assert t4_bright > t5_bright, "T4 should dominate for bright edge increments"
        
        detector.reset()
        print("Weights in detector:", detector.weights)
        print("T5 scale down:", detector._pathway_scales('down'))
        res_dark = detector.process_pair(frame1, frame2_dark)
        t4_dark = np.sum([v for v in res_dark.t4_maps.values()])
        t5_dark = np.sum([v for v in res_dark.t5_maps.values()])
        
        assert t5_dark > t4_dark, "T5 should dominate for dark edge decrements"

    def test_pathway_ablation(self, temp_connectome):
        """Phase 4: Ablation test showing T5 weights zeroed out eliminates dark-edge motion."""
        # Load and zero out T5 weights
        import json
        with open(temp_connectome, 'r') as f:
            data = json.load(f)
        
        # Zero out T5 weights
        weights_dict = data.get('weights', data)
        for k in weights_dict:
            if k.startswith('t5_'):
                weights_dict[k] = [0.0] * len(weights_dict[k])
        if 'weights' in data:
            data['weights'] = weights_dict
        else:
            data = weights_dict
                
        ablated_path = temp_connectome + "_ablated.json"
        with open(ablated_path, 'w') as f:
            json.dump(data, f)
            
        detector = MotionDetector(connectome_weights_path=ablated_path, tile_size=8)
        
        frame1 = np.full((32, 32), 128, dtype=np.uint8)
        frame2_dark = frame1.copy()
        frame2_dark[12:20, 12:20] = 0
        frame3_dark = frame1.copy()
        frame3_dark[14:22, 12:20] = 0
        frame4_dark = frame1.copy()
        frame4_dark[16:24, 12:20] = 0
        
        print("Weights in detector:", detector.weights)
        print("T5 scale down:", detector._pathway_scales('down'))
        detector.update(frame1)
        detector.update(frame2_dark)
        detector.update(frame3_dark)
        res_dark = detector.update(frame4_dark)
        
        # T5 is ablated, so dark-edge response should be zero
        t5_dark = np.sum([v for v in res_dark.t5_maps.values()])
        assert t5_dark == 0.0, "Ablated T5 pathway should produce zero response"
        
    def test_static_brightness(self, temp_connectome):
        """Phase 4: Static brightness produces negligible directional response."""
        detector = MotionDetector(connectome_weights_path=temp_connectome, tile_size=8)
        frame1 = np.full((32, 32), 100, dtype=np.uint8)
        frame2 = np.full((32, 32), 200, dtype=np.uint8)  # Sudden global brightness flash
        
        res = detector.process_pair(frame1, frame2)
        assert np.max(res.magnitude_map) < detector.config.motion_threshold, "Static brightness should produce negligible directional motion"

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
