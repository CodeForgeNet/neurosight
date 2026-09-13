"""
Phase 2: Core NeuroVision Math
Implements Hassenstein-Reichardt correlator using fly optic lobe weights.
L1/L2 → medulla → T4/T5 direction-selective neurons.
"""

from typing import Tuple, Optional, Dict, List, Any
import typing
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Motion detection using fly brain (T4/T5) architecture.
    Detects motion direction via ON (L1) and OFF (L2) pathways.
    """

    # Direction labels for T4/T5 subtypes
    DIRECTIONS = ['up', 'down', 'left', 'right']
    
    # Direction-to-pixel-offset mapping (for spatial correlation)
    # To detect motion in a direction, shift the current frame in the OPPOSITE direction
    DIRECTION_OFFSETS = {
        'up': (1, 0),       # dy, dx
        'down': (-1, 0),
        'left': (0, 1),
        'right': (0, -1)
    }

    # Direction to biological subtype mapping
    # T4/T5 tuning: a=down, b=front/right, c=up, d=back/left
    DIRECTION_SUBTYPES = {
        'down': 'a',
        'right': 'b',
        'up': 'c',
        'left': 'd'
    }

    def __init__(self, connectome_weights_path: str = "connectome_weights.json", tile_size: int = 8):
        """
        Initialize motion detector with biological weights.
        
        Args:
            connectome_weights_path: Path to connectome_weights.json (exported from Phase 1)
            tile_size: Spatial resolution of motion detection (8x8 or 16x16 pixels)
        """
        self.tile_size = tile_size
        self.connectome_weights = self._load_weights(connectome_weights_path)
        
        # Michaelis-Menten parameters for ON/OFF rectification (fit to real L1/L2 data)
        self.sigmoid_gain = 2.0   # Steepness of intensity response
        self.sigmoid_threshold = 0.5  # Half-saturation point (normalized intensity)
        
        logger.info(f"MotionDetector initialized (tile_size={tile_size})")

    def _load_weights(self, filepath: str) -> dict:
        """
        Load pre-computed connectome weights from JSON.
        
        Args:
            filepath: Path to connectome_weights.json
        
        Returns:
            {
                't4_up': [0.24, 0.18, 0.12, ...],
                't4_down': [...],
                ...
            }
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Read from "weights" key if it exists, otherwise fall back to raw data
            weights = data.get('weights', data)
            logger.info(f"Loaded connectome weights from {filepath}")
            return typing.cast(Dict[str, List[float]], weights)
        except FileNotFoundError:
            try:
                from .connectome_weights_data import WEIGHTS
                logger.info(f"Loaded connectome weights from python module fallback")
                weights = WEIGHTS.get('weights', WEIGHTS)
                return typing.cast(Dict[str, List[float]], weights)
            except ImportError:
                logger.error(f"Connectome weights not found at {filepath} and fallback module missing.")
                raise

    def _rectify_on(self, pixel_intensity: np.ndarray) -> np.ndarray:
        """
        ON pathway (L1): light-sensitive, increases with brightness.
        Michaelis-Menten-style rectification matching real L1 response.
        
        Args:
            pixel_intensity: Normalized intensity [0, 1]
        
        Returns:
            ON response [0, 1]
        """
        # Clip to [0, 1]
        intensity = np.clip(pixel_intensity, 0, 1)
        
        # Michaelis-Menten: response = intensity / (K + intensity)
        # With gain and threshold shift
        on_response = intensity / (self.sigmoid_threshold + intensity)
        
        return np.clip(on_response * self.sigmoid_gain, 0, 1).astype(np.float32)

    def _rectify_off(self, pixel_intensity: np.ndarray) -> np.ndarray:
        """
        OFF pathway (L2): dark-sensitive, increases when brightness decreases.
        
        Args:
            pixel_intensity: Normalized intensity [0, 1]
        
        Returns:
            OFF response [0, 1]
        """
        intensity = np.clip(pixel_intensity, 0, 1)
        
        # Inverted Michaelis-Menten for OFF cells
        off_response = (1.0 - intensity) / (self.sigmoid_threshold + (1.0 - intensity))
        
        return np.clip(off_response * self.sigmoid_gain, 0, 1).astype(np.float32)

    def _correlate_direction(
        self,
        on_prev: np.ndarray,
        on_curr: np.ndarray,
        off_prev: np.ndarray,
        off_curr: np.ndarray,
        direction: str
    ) -> Tuple[float, float]:
        """
        Hassenstein-Reichardt correlator: compute motion energy in one direction.
        
        Core equation: motion = delayed_signal_1 * current_signal_2
        Direction emerges from spatial offset + temporal delay.
        
        Args:
            on_prev, on_curr: ON pathway (L1) previous/current frame
            off_prev, off_curr: OFF pathway (L2) previous/current frame
            direction: 'up', 'down', 'left', 'right'
        
        Returns:
            (motion_magnitude, direction_confidence)
        """
        dy, dx = self.DIRECTION_OFFSETS[direction]
        
        # Spatial shift: "leading" pixel (where motion comes from)
        on_shifted = np.roll(on_curr, (dy, dx), axis=(0, 1))
        off_shifted = np.roll(off_curr, (dy, dx), axis=(0, 1))
        
        # Temporal correlation: delayed ON × current ON, delayed OFF × current OFF
        # (Fly brain uses neural delay lines; we approximate with frame difference)
        on_motion = on_prev * on_shifted  # Correlation arm 1
        off_motion = off_prev * off_shifted  # Correlation arm 2
        
        # T4 prefers ON (light flicker), T5 prefers OFF (dark flicker)
        t4_response = on_motion.mean()
        t5_response = off_motion.mean()
        
        # Get biological subtypes (a, b, c, d)
        subtype = self.DIRECTION_SUBTYPES.get(direction, 'a')
        
        t4_key = f"t4_{subtype}"
        t5_key = f"t5_{subtype}"
        
        if t4_key in self.connectome_weights and t5_key in self.connectome_weights:
            t4_weights = self.connectome_weights[t4_key]
            t5_weights = self.connectome_weights[t5_key]
            
            # Sum of primary ON-pathway medulla weights (Mi1, Mi4, Mi9, Tm3)
            # which are indices 0, 1, 2, 5 in our JSON array
            t4_scale = t4_weights[0] + t4_weights[1] + t4_weights[2] + t4_weights[5]
            
            # Sum of primary OFF-pathway medulla weights (Tm1, Tm2, Tm4, Tm9)
            # which are indices 3, 4, 6, 7 in our JSON array
            t5_scale = t5_weights[3] + t5_weights[4] + t5_weights[6] + t5_weights[7]
            
            # Combine pathways using biological scale
            total_scale = t4_scale + t5_scale + 1e-6
            motion_magnitude = (t4_response * t4_scale + t5_response * t5_scale) / total_scale
        else:
            motion_magnitude = (t4_response + t5_response) / 2.0
        
        # Confidence: how consistent is the response across the patch?
        # High standard deviation = noise/inconsistent motion across the patch.
        consistency = 1.0 - (on_motion.std() + off_motion.std()) / (t4_response + 1e-6)
        confidence = np.clip(consistency, 0, 1)
        
        return float(motion_magnitude), float(confidence)

    def detect_tile_motion(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        tile_y: int,
        tile_x: int
    ) -> Tuple[Optional[str], float]:
        """
        Detect motion direction in a single tile.
        
        Args:
            frame_prev, frame_curr: Grayscale frames (H, W)
            tile_y, tile_x: Tile top-left corner
        
        Returns:
            (best_direction, confidence)
            e.g., ('right', 0.87)
        """
        y_end = min(tile_y + self.tile_size, frame_curr.shape[0])
        x_end = min(tile_x + self.tile_size, frame_curr.shape[1])
        
        # Extract tile
        on_prev = self._rectify_on(frame_prev[tile_y:y_end, tile_x:x_end])
        on_curr = self._rectify_on(frame_curr[tile_y:y_end, tile_x:x_end])
        off_prev = self._rectify_off(frame_prev[tile_y:y_end, tile_x:x_end])
        off_curr = self._rectify_off(frame_curr[tile_y:y_end, tile_x:x_end])
        
        # Compute motion energy in all 4 directions
        motion_scores = {}
        for direction in self.DIRECTIONS:
            magnitude, confidence = self._correlate_direction(
                on_prev, on_curr, off_prev, off_curr, direction
            )
            motion_scores[direction] = (magnitude, confidence)
        
        # True motion requires opponent subtraction (e.g. Right - Left)
        # This matches how Lobula Plate Tangential Cells (LPTCs) integrate T4/T5 signals
        net_y = motion_scores['down'][0] - motion_scores['up'][0]
        net_x = motion_scores['right'][0] - motion_scores['left'][0]
        
        if abs(net_y) > abs(net_x):
            best_direction = 'down' if net_y > 0 else 'up'
            best_magnitude = abs(net_y)
            best_confidence = motion_scores[best_direction][1]
        else:
            best_direction = 'right' if net_x > 0 else 'left'
            best_magnitude = abs(net_x)
            best_confidence = motion_scores[best_direction][1]
        
        # No motion detected if net magnitude is near zero
        if best_magnitude < 0.01:
            return None, 0.0
        
        # Confidence correlates with the strength of the motion signal
        best_confidence = np.clip(best_magnitude, 0, 1)
        
        return best_direction, float(best_confidence)

    def process_frame(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a pair of consecutive frames, return saliency map + motion vectors.
        
        Args:
            frame_prev, frame_curr: Grayscale frames (H, W) or color (H, W, 3)
        
        Returns:
            (saliency_map, motion_vectors)
            - saliency_map: (H, W) heatmap of motion magnitude [0, 1]
            - motion_vectors: Dict of per-tile motion data
        """
        # Convert to grayscale if needed
        if frame_curr.ndim == 3:
            frame_prev = self._to_grayscale(frame_prev)
            frame_curr = self._to_grayscale(frame_curr)
        
        # Convert uint8 to float32 [0, 1] if needed
        if frame_prev.dtype == np.uint8:
            frame_prev = frame_prev.astype(np.float32) / 255.0
        if frame_curr.dtype == np.uint8:
            frame_curr = frame_curr.astype(np.float32) / 255.0
            
        H, W = frame_prev.shape[:2]
        
        # Tile the frame
        tiles_y = range(0, H, self.tile_size)
        tiles_x = range(0, W, self.tile_size)
        
        # Storage for per-tile motion
        motion_map = np.zeros((H, W), dtype=np.float32)
        motion_vectors = {}
        
        # Process each tile
        for ty in tiles_y:
            for tx in tiles_x:
                direction, confidence = self.detect_tile_motion(frame_prev, frame_curr, ty, tx)
                
                tile_key = f"({ty//self.tile_size}, {tx//self.tile_size})"
                motion_vectors[tile_key] = {
                    'direction': direction,
                    'confidence': confidence,
                    'y': ty,
                    'x': tx
                }
                
                # Fill motion map with confidence for saliency
                ye = min(ty + self.tile_size, H)
                xe = min(tx + self.tile_size, W)
                motion_map[ty:ye, tx:xe] = confidence
        
        # Upsample saliency map to frame resolution (smooth interpolation)
        saliency_map = self._upsample_saliency(motion_map, (H, W))
        
        return saliency_map, motion_vectors

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB/BGR to grayscale."""
        if frame.ndim == 3:
            res = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            return typing.cast(np.ndarray, res)
        return frame

    def _upsample_saliency(self, motion_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Upsample motion map to original frame resolution (bilinear interpolation).
        
        Args:
            motion_map: Low-res motion map (H/tile_size, W/tile_size)
            target_shape: Original frame shape (H, W)
        
        Returns:
            Upsampled saliency map (H, W)
        """
        from scipy.ndimage import zoom  # type: ignore
        
        zoom_factors = (target_shape[0] / motion_map.shape[0], target_shape[1] / motion_map.shape[1])
        saliency = zoom(motion_map, zoom_factors, order=1)  # Bilinear
        
        return np.clip(saliency, 0, 1).astype(np.float32)


if __name__ == "__main__":
    # Test detector initialization
    detector = MotionDetector(tile_size=8)
    print("MotionDetector ready")
