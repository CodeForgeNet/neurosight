"""
Phase 2: Core NeuroSight Math
Implements Hassenstein-Reichardt correlator using fly optic lobe weights.
L1/L2 → medulla → T4/T5 direction-selective neurons.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
import typing
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuroSightConfig:
    """Configuration for the lightweight motion detector core."""

    tile_size: int = 8
    motion_threshold: float = 0.01
    confidence_scale: float = 0.05
    input_color: str = "rgb"


@dataclass
class MotionResult:
    """Structured result returned by the newer motion API."""

    saliency_map: np.ndarray
    motion_vectors: Dict[str, Dict[str, Any]]
    motion_field: np.ndarray
    direction_maps: Dict[str, np.ndarray]
    confidence_map: np.ndarray
    magnitude_map: np.ndarray
    angle_map: np.ndarray
    global_vector: Tuple[float, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_legacy(self) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
        """Return the original `(saliency_map, motion_vectors)` API shape."""
        return self.saliency_map, self.motion_vectors


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

    def __init__(
        self,
        connectome_weights_path: str = "connectome_weights.json",
        tile_size: int = 8,
        config: Optional[NeuroSightConfig] = None,
    ):
        """
        Initialize motion detector with biological weights.
        
        Args:
            connectome_weights_path: Path to connectome_weights.json (exported from Phase 1)
            tile_size: Spatial resolution of motion detection (8x8 or 16x16 pixels)
        """
        self.config = config or NeuroSightConfig(tile_size=tile_size)
        self.tile_size = self.config.tile_size
        if self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        self.connectome_weights = self._load_weights(connectome_weights_path)
        self._tile_counts_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._shift_mask_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        
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
        
        # Spatial shift: "leading" pixel (where motion comes from).
        # Zero-padded shifting avoids artificial wraparound motion at tile edges.
        on_shifted = self._shift_no_wrap(on_curr, dy, dx)
        off_shifted = self._shift_no_wrap(off_curr, dy, dx)
        
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
        
        if (
            t4_key in self.connectome_weights
            and t5_key in self.connectome_weights
            and len(self.connectome_weights[t4_key]) >= 8
            and len(self.connectome_weights[t5_key]) >= 8
        ):
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
        
        # Confidence: how coherent is the response across the patch?
        # High spatial variability lowers confidence without discarding signal strength.
        response_mean = t4_response + t5_response
        response_variability = on_motion.std() + off_motion.std()
        consistency = response_mean / (response_mean + response_variability + 1e-6)
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
            best_consistency = motion_scores[best_direction][1]
        else:
            best_direction = 'right' if net_x > 0 else 'left'
            best_magnitude = abs(net_x)
            best_consistency = motion_scores[best_direction][1]
        
        # No motion detected if net magnitude is near zero
        if best_magnitude < self.config.motion_threshold:
            return None, 0.0
        
        # Confidence combines opponent motion strength with local response coherence.
        # The saturating strength term keeps weak-but-coherent motion visible while
        # still bounding very strong signals.
        normalized_strength = best_magnitude / (best_magnitude + self.config.confidence_scale)
        best_confidence = np.clip(normalized_strength * best_consistency, 0, 1)
        
        return best_direction, float(best_confidence)

    def process_pair(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray
    ) -> MotionResult:
        """
        Process a pair of consecutive frames, return saliency map + motion vectors.
        
        Args:
            frame_prev, frame_curr: Grayscale frames (H, W) or color (H, W, 3)
        
        Returns:
            MotionResult containing saliency, per-tile vectors, dense unitless
            motion field, directional maps, and diagnostics.
        """
        frame_prev, frame_curr = self._prepare_frame_pair(frame_prev, frame_curr)
        H, W = frame_prev.shape[:2]

        tile_direction, tile_confidence, direction_grids = self._detect_motion_grid(
            frame_prev,
            frame_curr,
        )
        saliency_map = self._expand_tile_grid(tile_confidence, H, W)
        motion_field = self._build_motion_field(tile_direction, tile_confidence, H, W)
        magnitude_map = np.linalg.norm(motion_field, axis=2).astype(np.float32)
        angle_map = np.arctan2(motion_field[..., 1], motion_field[..., 0]).astype(np.float32)
        angle_map = np.where(magnitude_map > 0, angle_map, 0.0).astype(np.float32)
        direction_maps = {
            direction: self._expand_tile_grid(direction_grids[direction], H, W)
            for direction in self.DIRECTIONS
        }
        motion_vectors = self._build_motion_vectors(tile_direction, tile_confidence, H, W)
        valid_tile_count = int(np.count_nonzero(tile_direction >= 0))
        
        active = saliency_map > 0
        if np.any(active):
            global_dx = float(np.mean(motion_field[..., 0][active]))
            global_dy = float(np.mean(motion_field[..., 1][active]))
        else:
            global_dx = 0.0
            global_dy = 0.0

        diagnostics = {
            'tile_size': self.tile_size,
            'frame_shape': (H, W),
            'total_tiles': len(motion_vectors),
            'active_tiles': valid_tile_count,
            'motion_threshold': self.config.motion_threshold,
        }

        return MotionResult(
            saliency_map=saliency_map,
            motion_vectors=motion_vectors,
            motion_field=motion_field,
            direction_maps=direction_maps,
            confidence_map=saliency_map.copy(),
            magnitude_map=magnitude_map,
            angle_map=angle_map,
            global_vector=(global_dx, global_dy),
            diagnostics=diagnostics,
        )

    def process_frame(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a pair of consecutive frames, return saliency map + motion vectors.

        This preserves the original public API. New code should prefer
        `process_pair()` for the structured MotionResult.
        """
        return self.process_pair(frame_prev, frame_curr).as_legacy()

    def _detect_motion_grid(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Compute best direction and confidence for every tile in vectorized batches."""
        on_prev = self._rectify_on(frame_prev)
        on_curr = self._rectify_on(frame_curr)
        off_prev = self._rectify_off(frame_prev)
        off_curr = self._rectify_off(frame_curr)

        magnitudes = {}
        consistencies = {}
        for direction in self.DIRECTIONS:
            magnitudes[direction], consistencies[direction] = self._correlate_direction_grid(
                on_prev,
                on_curr,
                off_prev,
                off_curr,
                direction,
            )

        net_y = magnitudes['down'] - magnitudes['up']
        net_x = magnitudes['right'] - magnitudes['left']
        use_y = np.abs(net_y) > np.abs(net_x)

        tile_direction = np.full(net_x.shape, -1, dtype=np.int8)
        tile_direction = np.where(use_y & (net_y > 0), 1, tile_direction)
        tile_direction = np.where(use_y & (net_y <= 0), 0, tile_direction)
        tile_direction = np.where(~use_y & (net_x > 0), 3, tile_direction)
        tile_direction = np.where(~use_y & (net_x <= 0), 2, tile_direction)

        best_magnitude = np.where(use_y, np.abs(net_y), np.abs(net_x)).astype(np.float32)
        best_consistency = np.zeros_like(best_magnitude, dtype=np.float32)
        direction_to_index = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        for direction, index in direction_to_index.items():
            best_consistency = np.where(
                tile_direction == index,
                consistencies[direction],
                best_consistency,
            )

        active = best_magnitude >= self.config.motion_threshold
        tile_direction = np.where(active, tile_direction, -1).astype(np.int8)
        normalized_strength = best_magnitude / (
            best_magnitude + self.config.confidence_scale
        )
        tile_confidence = np.where(
            active,
            normalized_strength * best_consistency,
            0.0,
        )
        tile_confidence = np.clip(tile_confidence, 0.0, 1.0).astype(np.float32)

        direction_grids = {
            'up': np.where(tile_direction == 0, tile_confidence, 0.0).astype(np.float32),
            'down': np.where(tile_direction == 1, tile_confidence, 0.0).astype(np.float32),
            'left': np.where(tile_direction == 2, tile_confidence, 0.0).astype(np.float32),
            'right': np.where(tile_direction == 3, tile_confidence, 0.0).astype(np.float32),
        }

        return tile_direction, tile_confidence, direction_grids

    def _correlate_direction_grid(
        self,
        on_prev: np.ndarray,
        on_curr: np.ndarray,
        off_prev: np.ndarray,
        off_curr: np.ndarray,
        direction: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute motion magnitude and coherence for one direction over all tiles."""
        dy, dx = self.DIRECTION_OFFSETS[direction]
        on_shifted = self._shift_no_wrap(on_curr, dy, dx)
        off_shifted = self._shift_no_wrap(off_curr, dy, dx)
        valid_shift = self._same_tile_shift_mask(on_curr.shape, dy, dx)

        on_motion = on_prev * on_shifted * valid_shift
        off_motion = off_prev * off_shifted * valid_shift

        t4_response, on_std = self._tile_mean_and_std(on_motion)
        t5_response, off_std = self._tile_mean_and_std(off_motion)
        t4_scale, t5_scale = self._pathway_scales(direction)

        total_scale = t4_scale + t5_scale + 1e-6
        magnitude = (
            (t4_response * t4_scale + t5_response * t5_scale) / total_scale
        ).astype(np.float32)

        response_mean = t4_response + t5_response
        response_variability = on_std + off_std
        consistency = response_mean / (response_mean + response_variability + 1e-6)

        return magnitude, np.clip(consistency, 0.0, 1.0).astype(np.float32)

    def _pathway_scales(self, direction: str) -> Tuple[float, float]:
        """Return T4/T5 direction scales, falling back for legacy four-value fixtures."""
        subtype = self.DIRECTION_SUBTYPES.get(direction, 'a')
        t4_weights = self.connectome_weights.get(f"t4_{subtype}")
        t5_weights = self.connectome_weights.get(f"t5_{subtype}")

        if t4_weights is not None and t5_weights is not None:
            if len(t4_weights) >= 8 and len(t5_weights) >= 8:
                t4_scale = t4_weights[0] + t4_weights[1] + t4_weights[2] + t4_weights[5]
                t5_scale = t5_weights[3] + t5_weights[4] + t5_weights[6] + t5_weights[7]
                return float(t4_scale), float(t5_scale)

        return 1.0, 1.0

    def _tile_mean_and_std(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-tile mean and std for a frame-sized 2D array."""
        H, W = values.shape[:2]
        tile_size = self.tile_size
        tiles_h = (H + tile_size - 1) // tile_size
        tiles_w = (W + tile_size - 1) // tile_size
        padded_h = tiles_h * tile_size
        padded_w = tiles_w * tile_size

        padded = np.zeros((padded_h, padded_w), dtype=np.float32)
        padded[:H, :W] = values
        blocks = padded.reshape(tiles_h, tile_size, tiles_w, tile_size)
        sums = blocks.sum(axis=(1, 3))
        square_sums = (blocks * blocks).sum(axis=(1, 3))

        counts = self._tile_pixel_counts(H, W)
        means = sums / counts
        variances = np.maximum(square_sums / counts - means * means, 0.0)
        return means.astype(np.float32), np.sqrt(variances).astype(np.float32)

    def _tile_pixel_counts(self, height: int, width: int) -> np.ndarray:
        """Return actual pixel counts for full and partial edge tiles."""
        cache_key = (height, width)
        if cache_key in self._tile_counts_cache:
            return self._tile_counts_cache[cache_key]

        tile_size = self.tile_size
        y_starts = np.arange(0, height, tile_size)
        x_starts = np.arange(0, width, tile_size)
        y_counts = np.minimum(tile_size, height - y_starts).astype(np.float32)
        x_counts = np.minimum(tile_size, width - x_starts).astype(np.float32)
        counts = y_counts[:, None] * x_counts[None, :]
        self._tile_counts_cache[cache_key] = counts
        return counts

    def _same_tile_shift_mask(
        self,
        shape: Tuple[int, int],
        dy: int,
        dx: int
    ) -> np.ndarray:
        """Mask shifted pixels whose source stays inside the same detector tile."""
        height, width = shape
        cache_key = (height, width, dy, dx)
        if cache_key in self._shift_mask_cache:
            return self._shift_mask_cache[cache_key]

        y_tiles = (np.arange(height) // self.tile_size)[:, None]
        x_tiles = (np.arange(width) // self.tile_size)[None, :]
        tile_ids = (y_tiles * ((width + self.tile_size - 1) // self.tile_size) + x_tiles + 1)
        shifted_tile_ids = self._shift_no_wrap(tile_ids.astype(np.int32), dy, dx)
        mask = (shifted_tile_ids == tile_ids).astype(np.float32)
        self._shift_mask_cache[cache_key] = mask
        return mask

    def _expand_tile_grid(self, tile_grid: np.ndarray, height: int, width: int) -> np.ndarray:
        """Paint a tile grid back to frame resolution."""
        expanded = np.repeat(
            np.repeat(tile_grid, self.tile_size, axis=0),
            self.tile_size,
            axis=1,
        )
        return expanded[:height, :width].astype(np.float32, copy=False)

    def _build_motion_field(
        self,
        tile_direction: np.ndarray,
        tile_confidence: np.ndarray,
        height: int,
        width: int
    ) -> np.ndarray:
        """Build a frame-sized unitless vector field from tile directions."""
        tile_dx = np.zeros(tile_direction.shape, dtype=np.float32)
        tile_dy = np.zeros(tile_direction.shape, dtype=np.float32)

        tile_dy = np.where(tile_direction == 0, -tile_confidence, tile_dy)
        tile_dy = np.where(tile_direction == 1, tile_confidence, tile_dy)
        tile_dx = np.where(tile_direction == 2, -tile_confidence, tile_dx)
        tile_dx = np.where(tile_direction == 3, tile_confidence, tile_dx)

        motion_field = np.zeros((height, width, 2), dtype=np.float32)
        motion_field[..., 0] = self._expand_tile_grid(tile_dx, height, width)
        motion_field[..., 1] = self._expand_tile_grid(tile_dy, height, width)
        return motion_field

    def _build_motion_vectors(
        self,
        tile_direction: np.ndarray,
        tile_confidence: np.ndarray,
        height: int,
        width: int
    ) -> Dict[str, Dict[str, Any]]:
        """Build the legacy per-tile vector dictionary from vectorized tile grids."""
        labels = ['up', 'down', 'left', 'right']
        vectors = {}
        tiles_h, tiles_w = tile_direction.shape
        for tile_y in range(tiles_h):
            for tile_x in range(tiles_w):
                direction_index = int(tile_direction[tile_y, tile_x])
                direction = labels[direction_index] if direction_index >= 0 else None
                vectors[f"({tile_y}, {tile_x})"] = {
                    'direction': direction,
                    'confidence': float(tile_confidence[tile_y, tile_x]),
                    'y': min(tile_y * self.tile_size, height),
                    'x': min(tile_x * self.tile_size, width),
                }
        return vectors

    def _prepare_frame_pair(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate frames, convert to grayscale, and normalize to float32 [0, 1]."""
        if frame_prev is None or frame_curr is None:
            raise ValueError("frame_prev and frame_curr must be numpy arrays")

        prev = np.asarray(frame_prev)
        curr = np.asarray(frame_curr)
        if prev.ndim not in (2, 3) or curr.ndim not in (2, 3):
            raise ValueError("frames must be 2D grayscale or 3D color arrays")

        if prev.ndim == 3:
            prev = self._to_grayscale(prev)
        if curr.ndim == 3:
            curr = self._to_grayscale(curr)

        if prev.shape != curr.shape:
            raise ValueError(
                f"frame shapes must match after grayscale conversion: {prev.shape} != {curr.shape}"
            )
        if prev.size == 0:
            raise ValueError("frames must not be empty")

        return self._normalize_frame(prev), self._normalize_frame(curr)

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB/BGR to grayscale."""
        if frame.ndim == 3:
            if frame.shape[2] < 3:
                raise ValueError("color frames must have at least 3 channels")
            if self.config.input_color not in ("rgb", "bgr"):
                raise ValueError("input_color must be either 'rgb' or 'bgr'")
            weights = [0.299, 0.587, 0.114]
            if self.config.input_color == "bgr":
                weights = [0.114, 0.587, 0.299]
            res = np.dot(frame[..., :3], weights)
            return typing.cast(np.ndarray, res)
        return frame

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert integer or float frames into float32 intensity in [0, 1]."""
        if np.issubdtype(frame.dtype, np.integer):
            info = np.iinfo(frame.dtype)
            normalized = frame.astype(np.float32) / float(info.max)
        else:
            normalized = frame.astype(np.float32, copy=False)
            if normalized.size and float(np.nanmax(normalized)) > 1.0:
                normalized = normalized / 255.0
        return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _shift_no_wrap(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Shift a 2D array with zero padding instead of toroidal wraparound."""
        shifted = np.zeros_like(frame)
        height, width = frame.shape[:2]

        src_y0 = max(0, -dy)
        src_y1 = height - max(0, dy)
        dst_y0 = max(0, dy)
        dst_y1 = height - max(0, -dy)
        src_x0 = max(0, -dx)
        src_x1 = width - max(0, dx)
        dst_x0 = max(0, dx)
        dst_x1 = width - max(0, -dx)

        if src_y0 >= src_y1 or src_x0 >= src_x1:
            return shifted

        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
        return shifted

    @staticmethod
    def _direction_to_unit_vector(direction: str) -> Tuple[float, float]:
        vectors = {
            'up': (0.0, -1.0),
            'down': (0.0, 1.0),
            'left': (-1.0, 0.0),
            'right': (1.0, 0.0),
        }
        return vectors[direction]

    def _upsample_saliency(self, motion_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Return a frame-resolution saliency map.
        
        Args:
            motion_map: Motion map already painted at frame resolution
            target_shape: Original frame shape (H, W)
        
        Returns:
            Saliency map (H, W)
        """
        if motion_map.shape != target_shape:
            raise ValueError(
                f"motion_map shape must match target_shape: {motion_map.shape} != {target_shape}"
            )
        
        return np.clip(motion_map, 0, 1).astype(np.float32, copy=False)


if __name__ == "__main__":
    # Test detector initialization
    detector = MotionDetector(tile_size=8)
    print("MotionDetector ready")
