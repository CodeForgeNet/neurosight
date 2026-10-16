"""
Core NeuroSight math.

Implements a Hassenstein-Reichardt correlator using fly optic lobe weights
(L1/L2 -> medulla -> T4/T5 direction-selective neurons).
"""

import json
import logging
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuroSightConfig:
    """Configuration for the lightweight motion detector core."""

    tile_size: int = 8
    motion_threshold: float = 0.01
    confidence_scale: float = 0.05
    input_color: str = "rgb"
    scales: Tuple[int, ...] = (1, 2)
    tau_lowpass: float = 0.05
    default_dt: float = 0.033
    contrast_normalization: bool = True
    lateral_inhibition: bool = True
    opponent_subtraction: bool = True
    adaptation_rate: float = 0.01
    
    # Real-world robustness
    temporal_denoising: bool = True
    denoising_alpha: float = 0.5
    flicker_suppression: bool = True
    ego_motion_compensation: bool = True
    blur_handling: bool = False
    
    # Robotics outputs
    enable_regions: bool = True
    enable_looming: bool = True
    enable_steering: bool = True
    include_pathway_maps: bool = True

    def __post_init__(self):
        """Validate configuration values eagerly with clear messages."""
        if not isinstance(self.tile_size, int) or self.tile_size <= 0:
            raise ValueError(f"tile_size must be a positive integer, got {self.tile_size!r}")
        if not self.scales or any(
            not isinstance(s, int) or s <= 0 for s in self.scales
        ):
            raise ValueError(f"scales must be a non-empty tuple of positive ints, got {self.scales!r}")
        if self.motion_threshold < 0:
            raise ValueError(f"motion_threshold must be >= 0, got {self.motion_threshold!r}")
        if self.confidence_scale <= 0:
            raise ValueError(f"confidence_scale must be > 0, got {self.confidence_scale!r}")
        if self.tau_lowpass <= 0:
            raise ValueError(f"tau_lowpass must be > 0, got {self.tau_lowpass!r}")
        if self.default_dt <= 0:
            raise ValueError(f"default_dt must be > 0, got {self.default_dt!r}")
        if self.input_color not in ("rgb", "bgr"):
            raise ValueError(f"input_color must be 'rgb' or 'bgr', got {self.input_color!r}")
        if not 0.0 <= self.adaptation_rate <= 1.0:
            raise ValueError(f"adaptation_rate must be within [0, 1], got {self.adaptation_rate!r}")
        if not 0.0 <= self.denoising_alpha <= 1.0:
            raise ValueError(f"denoising_alpha must be within [0, 1], got {self.denoising_alpha!r}")

    @classmethod
    def preset_education(cls, **kwargs):
        """Slower, highly explainable settings with default robustness disabled."""
        defaults = dict(
            temporal_denoising=False,
            flicker_suppression=False,
            ego_motion_compensation=False,
            blur_handling=False,
            enable_regions=False,
            enable_looming=False,
            enable_steering=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def preset_edge_fast(cls, **kwargs):
        """Minimal latency settings optimized for Raspberry Pi, turning off some heavy robotics processing."""
        defaults = dict(
            scales=(1,),
            blur_handling=False,
            temporal_denoising=False,
            ego_motion_compensation=False,
            enable_looming=False,
            include_pathway_maps=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def preset_robust(cls, **kwargs):
        """High noise suppression for complex lighting or bumpy movement."""
        defaults = dict(
            temporal_denoising=True,
            denoising_alpha=0.3,
            flicker_suppression=True,
            ego_motion_compensation=True,
            blur_handling=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def preset_looming(cls, **kwargs):
        """Settings optimized for collision warning detection."""
        defaults = dict(
            ego_motion_compensation=True,
            enable_looming=True,
            enable_steering=True,
            enable_regions=True,
            scales=(1, 2, 4),
        )
        defaults.update(kwargs)
        return cls(**defaults)

@dataclass
class MotionResult:
    """Structured result returned by the newer motion API.

    The frame-resolution direction and pathway maps are expanded lazily from
    their tile-resolution grids, so constructing a result is cheap and those
    arrays are only allocated when a caller actually reads them.
    """

    saliency_map: np.ndarray
    motion_field: np.ndarray
    confidence_map: np.ndarray
    magnitude_map: np.ndarray
    angle_map: np.ndarray
    global_vector: Tuple[float, float]
    
    # Ego-motion / robustness
    ego_motion_vector: Optional[Tuple[float, float]] = None
    ego_compensated_field: Optional[np.ndarray] = None
    scene_quality: Optional[Dict[str, float]] = None
    # Robotics outputs
    regions: Optional[List[Dict[str, Any]]] = None
    looming_score: Optional[float] = None
    time_to_contact: Optional[float] = None
    steering_signal: Optional[Dict[str, float]] = None

    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Tile-resolution grids backing the lazy map properties, plus caches.
    _tile_direction: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _tile_confidence: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _tile_size: int = field(default=8, repr=False, compare=False)
    _frame_shape: Optional[Tuple[int, int]] = field(default=None, repr=False, compare=False)
    _direction_grids: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False, compare=False)
    _t4_grids: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False, compare=False)
    _t5_grids: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False, compare=False)
    _include_pathway: bool = field(default=False, repr=False, compare=False)
    _motion_vectors_cache: Optional[Dict[str, Dict[str, Any]]] = field(default=None, repr=False, compare=False)
    _direction_maps_cache: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False, compare=False)
    _t4_maps_cache: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False, compare=False)
    _t5_maps_cache: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False, compare=False)

    @staticmethod
    def _expand(grid: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Paint a tile-resolution grid back to frame resolution."""
        height, width = shape
        return cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    def _expand_grids(
        self,
        grids: Optional[Dict[str, np.ndarray]],
        cache: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        if cache is not None:
            return cache
        if grids is None or self._frame_shape is None:
            return {}
        return {
            direction: self._expand(grid, self._frame_shape)
            for direction, grid in grids.items()
        }

    @property
    def direction_maps(self) -> Dict[str, np.ndarray]:
        """Per-direction confidence grids at frame resolution (lazy)."""
        maps = self._expand_grids(self._direction_grids, self._direction_maps_cache)
        if maps:
            self._direction_maps_cache = maps
        return maps

    @property
    def t4_maps(self) -> Dict[str, np.ndarray]:
        """ON-pathway (T4) response grids at frame resolution (lazy)."""
        if not self._include_pathway:
            return {}
        maps = self._expand_grids(self._t4_grids, self._t4_maps_cache)
        if maps:
            self._t4_maps_cache = maps
        return maps

    @property
    def t5_maps(self) -> Dict[str, np.ndarray]:
        """OFF-pathway (T5) response grids at frame resolution (lazy)."""
        if not self._include_pathway:
            return {}
        maps = self._expand_grids(self._t5_grids, self._t5_maps_cache)
        if maps:
            self._t5_maps_cache = maps
        return maps

    @property
    def motion_vectors(self) -> Dict[str, Dict[str, Any]]:
        """Lazy evaluation of the legacy motion_vectors dictionary."""
        if self._motion_vectors_cache is not None:
            return self._motion_vectors_cache
            
        if self._tile_direction is None or self._tile_confidence is None or self._frame_shape is None:
            return {}

        labels = ['up', 'down', 'left', 'right']
        vectors = {}
        height, width = self._frame_shape
        tiles_h, tiles_w = self._tile_direction.shape
        
        for tile_y in range(tiles_h):
            for tile_x in range(tiles_w):
                direction_index = int(self._tile_direction[tile_y, tile_x])
                direction = labels[direction_index] if direction_index >= 0 else None
                vectors[f"({tile_y}, {tile_x})"] = {
                    'direction': direction,
                    'confidence': float(self._tile_confidence[tile_y, tile_x]),
                    'y': min(tile_y * self._tile_size, height),
                    'x': min(tile_x * self._tile_size, width),
                }
                
        # We can't mutate self in a frozen dataclass, but MotionResult is not frozen.
        self._motion_vectors_cache = vectors
        return vectors

    def as_legacy(self) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
        """Return the original `(saliency_map, motion_vectors)` API shape."""
        return self.saliency_map, self.motion_vectors


class MotionDetector:
    """
    Motion detection using fly brain (T4/T5) architecture.
    Detects motion direction via ON (L1) and OFF (L2) pathways.

    This detector is **stateful**: :meth:`update` mutates internal temporal state
    (frame history, adaptation, ego-motion estimate) and is therefore **not
    thread-safe**. Use one instance per stream/thread. For two-frame, stateless
    use, call :meth:`process_pair`.
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

    # Michaelis-Menten rectification parameters (fit to real L1/L2 data)
    DEFAULT_SIGMOID_GAIN = 2.0       # Steepness of intensity response
    DEFAULT_SIGMOID_THRESHOLD = 0.5  # Half-saturation point (normalized intensity)

    # Maximum number of cached per-shape shift masks (bounds memory when the
    # input resolution varies over the lifetime of a detector).
    SHIFT_MASK_CACHE_SIZE = 64

    def __init__(
        self,
        connectome_weights_path: Optional[str] = None,
        tile_size: int = 8,
        config: Optional[NeuroSightConfig] = None,
    ):
        """
        Initialize motion detector with biological weights.

        Args:
            connectome_weights_path: Optional path to a ``connectome_weights.json``
                file. When ``None`` (default) the weights bundled with the package
                are used, so no external file is required.
            tile_size: Spatial resolution of motion detection (8x8 or 16x16 pixels)
        """
        self.config = config or NeuroSightConfig(tile_size=tile_size)
        self.tile_size = self.config.tile_size
        if self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        self.connectome_weights = self._load_weights(connectome_weights_path)
        self.inputs_order = self.connectome_weights.get('inputs_order', ["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9"])
        self.weights = self.connectome_weights.get('weights', self.connectome_weights)
        self._shift_mask_cache: "OrderedDict[Tuple[int, int, int, int], np.ndarray]" = OrderedDict()
        self._pathway_scale_cache: Dict[str, Tuple[float, float]] = {}
        
        self.sigmoid_gain = self.DEFAULT_SIGMOID_GAIN
        self.sigmoid_threshold = self.DEFAULT_SIGMOID_THRESHOLD
        
        self.reset()
        logger.info(f"MotionDetector initialized (tile_size={tile_size})")

    def _fallback_weights(self) -> dict:
        """Load the weights that ship inside the package."""
        try:
            from .connectome_weights_data import WEIGHTS
        except ImportError as exc:  # pragma: no cover - packaging regression guard
            raise FileNotFoundError(
                "Connectome weights were not found on disk and the bundled "
                "fallback module 'neurosight.connectome_weights_data' is missing. "
                "Reinstall the package or pass a valid connectome_weights_path."
            ) from exc
        logger.info("Loaded connectome weights from bundled python module")
        return WEIGHTS

    @classmethod
    def _validate_weights(cls, data: Any, source: str) -> dict:
        """Validate the structure of a connectome weights payload."""
        if not isinstance(data, dict):
            raise ValueError(
                f"Connectome weights from {source} must be a JSON object, "
                f"got {type(data).__name__}."
            )

        weights = data.get('weights', data)
        if not isinstance(weights, dict) or not weights:
            raise ValueError(
                f"Connectome weights from {source} contain no 'weights' mapping."
            )

        for key, values in weights.items():
            if key == 'inputs_order':
                continue
            if not isinstance(values, (list, tuple)):
                raise ValueError(
                    f"Connectome weight '{key}' from {source} must be a list of numbers."
                )
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(
                    f"Connectome weight '{key}' from {source} must be a non-empty 1D list."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Connectome weight '{key}' from {source} contains non-finite values."
                )
        return data

    def _load_weights(self, filepath: Optional[str]) -> dict:
        """Load and validate connectome weights, falling back to bundled data."""
        if not filepath:
            return self._fallback_weights()

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.info(
                "Connectome weights not found at %s; using bundled fallback.", filepath
            )
            return self._fallback_weights()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Connectome weights file '{filepath}' is not valid JSON: {exc}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Could not read connectome weights file '{filepath}': {exc}"
            ) from exc

        logger.info(f"Loaded connectome weights from {filepath}")
        return self._validate_weights(data, filepath)

    def _rectify_on(self, contrast: np.ndarray) -> np.ndarray:
        # Responds to positive contrast (bright edges)
        return (contrast / (self.sigmoid_threshold + contrast + 1e-6)) * self.sigmoid_gain

    def _rectify_off(self, contrast: np.ndarray) -> np.ndarray:
        # Responds to negative contrast (dark edges), passed in as positive absolute contrast
        return (contrast / (self.sigmoid_threshold + contrast + 1e-6)) * self.sigmoid_gain

    def reset(self):
        """Reset temporal state for a new video or sequence."""
        self.sigmoid_gain = self.DEFAULT_SIGMOID_GAIN
        self.sigmoid_threshold = self.DEFAULT_SIGMOID_THRESHOLD
        self._last_frame: Optional[np.ndarray] = None
        self._on_lowpass: Optional[np.ndarray] = None
        self._off_lowpass: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        
        # Real-world robustness state
        self._smoothed_frame: Optional[np.ndarray] = None
        self._ego_motion_vector: Optional[Tuple[float, float]] = None

    def update(
        self,
        frame_curr: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Optional[MotionResult]:
        """
        Process a new frame and update state, returning a MotionResult.
        If this is the first frame or resolution changes, state resets and returns None.
        """
        if frame_curr is None:
            raise ValueError("frame_curr must be a numpy array")
            
        curr = np.asarray(frame_curr)
        if curr.ndim == 3:
            curr = self._to_grayscale(curr)
        
        curr = self._normalize_frame(curr)
        if self.config.blur_handling:
            curr = cv2.GaussianBlur(curr, (3, 3), 0).astype(np.float32)

        if self.config.temporal_denoising:
            if self._smoothed_frame is None:
                self._smoothed_frame = curr.copy()
            else:
                self._smoothed_frame = (
                    self.config.denoising_alpha * curr +
                    (1.0 - self.config.denoising_alpha) * self._smoothed_frame
                )
            process_frame = self._smoothed_frame
        else:
            process_frame = curr

        H, W = curr.shape[:2]
        
        if self._last_frame_shape is not None and self._last_frame_shape != (H, W):
            logger.info("Resolution changed, resetting temporal state.")
            self.reset()
            
        self._last_frame_shape = (H, W)
        
        if self._last_frame is None:
            self._last_frame = process_frame.copy()
            self._last_timestamp = timestamp
            self._on_lowpass = np.zeros_like(process_frame)
            self._off_lowpass = np.zeros_like(process_frame)
            return None
            
        delta = curr - self._last_frame

        if self.config.flicker_suppression:
            delta = delta - float(np.median(delta))
        
        if self.config.contrast_normalization:
            # Luminance gain control. Floor the denominator so near-black
            # regions cannot produce explosive gain, then clip to a stable range.
            denominator = np.maximum(self._last_frame, 0.05)
            delta = np.clip(delta / denominator, -1.0, 1.0)
            
        if self.config.lateral_inhibition:
            local_mean, _ = self._tile_mean_and_std(delta)
            expanded_mean = self._expand_tile_grid(local_mean, H, W)
            delta = delta - expanded_mean
            
        self._last_frame = process_frame.copy()
        
        on_curr = self._rectify_on(np.clip(delta, 0, 1)).astype(np.float32)
        off_curr = self._rectify_off(np.clip(-delta, 0, 1)).astype(np.float32)
        
        mean_abs_contrast = float(np.mean(np.abs(delta)))
        self.sigmoid_threshold = (
            (1.0 - self.config.adaptation_rate) * self.sigmoid_threshold +
            self.config.adaptation_rate * mean_abs_contrast
        )
        self.sigmoid_threshold = np.clip(self.sigmoid_threshold, 0.05, 0.5)

        if timestamp is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt <= 0:
                dt = self.config.default_dt
        else:
            dt = self.config.default_dt
            
        self._last_timestamp = timestamp
        
        alpha_lp = 1.0 - np.exp(-dt / max(self.config.tau_lowpass, 1e-6))
        alpha_lp = np.clip(alpha_lp, 0.0, 1.0)
        
        self._on_lowpass = (1.0 - alpha_lp) * self._on_lowpass + alpha_lp * on_curr
        self._off_lowpass = (1.0 - alpha_lp) * self._off_lowpass + alpha_lp * off_curr
        
        on_highpass = on_curr - self._on_lowpass
        off_highpass = off_curr - self._off_lowpass
        
        tile_direction, tile_confidence, direction_grids, t4_grids, t5_grids = self._detect_motion_grid_stateful(
            self._on_lowpass, on_highpass,
            self._off_lowpass, off_highpass
        )
        
        motion_field = self._build_motion_field(tile_direction, tile_confidence, H, W)

        # Build final saliency and magnitude map
        saliency_map = self._expand_tile_grid(tile_confidence, H, W)
        magnitude_map = np.linalg.norm(motion_field, axis=2).astype(np.float32)
        angle_map = np.arctan2(motion_field[..., 1], motion_field[..., 0]).astype(np.float32)
        angle_map = np.where(magnitude_map > 0, angle_map, 0.0).astype(np.float32)
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
            'total_tiles': tile_direction.size,
            'active_tiles': valid_tile_count,
            'motion_threshold': self.config.motion_threshold,
            'scales': self.config.scales,
            'dt': dt
        }

        ego_vector = (global_dx, global_dy)
        ego_compensated_field = motion_field.copy()

        if self.config.ego_motion_compensation:
            alpha_ego = self.config.adaptation_rate
            if self._ego_motion_vector is None:
                self._ego_motion_vector = ego_vector
            else:
                self._ego_motion_vector = (
                    self._ego_motion_vector[0] * (1 - alpha_ego) + ego_vector[0] * alpha_ego,
                    self._ego_motion_vector[1] * (1 - alpha_ego) + ego_vector[1] * alpha_ego
                )
            
            ego_dx, ego_dy = self._ego_motion_vector
            ego_compensated_field[..., 0] = np.where(active, ego_compensated_field[..., 0] - ego_dx, 0.0)
            ego_compensated_field[..., 1] = np.where(active, ego_compensated_field[..., 1] - ego_dy, 0.0)
            
        scene_quality = {
            'exposure_clip': float(np.mean((curr < 0.05) | (curr > 0.95))),
            'global_contrast': float(np.std(curr)),
        }
        
        # Robotics outputs
        regions = self._extract_regions(saliency_map, ego_compensated_field)
        looming_score, ttc = self._compute_looming_and_ttc(ego_compensated_field, saliency_map, dt=dt)
        steering_signal = self._compute_steering_signal(saliency_map)

        return MotionResult(
            saliency_map=saliency_map,
            motion_field=motion_field,
            confidence_map=saliency_map.copy(),
            magnitude_map=magnitude_map,
            angle_map=angle_map,
            global_vector=(global_dx, global_dy),
            ego_motion_vector=self._ego_motion_vector,
            ego_compensated_field=ego_compensated_field,
            scene_quality=scene_quality,
            regions=regions,
            looming_score=looming_score,
            time_to_contact=ttc,
            steering_signal=steering_signal,
            diagnostics=diagnostics,
            _direction_grids=direction_grids,
            _t4_grids=t4_grids,
            _t5_grids=t5_grids,
            _include_pathway=self.config.include_pathway_maps,
            _tile_direction=tile_direction,
            _tile_confidence=tile_confidence,
            _tile_size=self.tile_size,
            _frame_shape=(H, W)
        )

    def process_pair(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        timestamp: Optional[float] = None
    ) -> MotionResult:
        """
        Process a pair of consecutive frames without mutating streaming state.

        Signal convention (important): this stateless path correlates the
        *rectified raw intensity* of the previous frame against the spatially
        shifted current frame (a two-sample Reichardt correlator). The stateful
        :meth:`update` path instead correlates *rectified temporal contrast*
        through an adaptive low/high-pass filter. Both are Reichardt-style
        correlators and are expected to agree on direction, but they are not
        numerically identical: the pair path has no temporal adaptation, delay
        line, or warm-up, so magnitudes/confidences will differ.

        Use this when you have exactly two frames and want an immediate result;
        use :meth:`update` for continuous streams where temporal adaptation
        matters.
        """
        frame_prev, frame_curr = self._prepare_frame_pair(frame_prev, frame_curr)
        if self.config.blur_handling:
            frame_prev = cv2.GaussianBlur(frame_prev, (3, 3), 0).astype(np.float32)
            frame_curr = cv2.GaussianBlur(frame_curr, (3, 3), 0).astype(np.float32)

        height, width = frame_prev.shape[:2]
        tile_direction, tile_confidence, direction_grids, t4_grids, t5_grids = (
            self._detect_motion_grid_pair(frame_prev, frame_curr)
        )

        return self._build_result(
            tile_direction=tile_direction,
            tile_confidence=tile_confidence,
            direction_grids=direction_grids,
            t4_grids=t4_grids,
            t5_grids=t5_grids,
            height=height,
            width=width,
            source_frame=frame_curr,
            diagnostics_extra={'mode': 'pair'},
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

    def _build_result(
        self,
        tile_direction: np.ndarray,
        tile_confidence: np.ndarray,
        direction_grids: Dict[str, np.ndarray],
        t4_grids: Dict[str, np.ndarray],
        t5_grids: Dict[str, np.ndarray],
        height: int,
        width: int,
        source_frame: np.ndarray,
        diagnostics_extra: Optional[Dict[str, Any]] = None,
    ) -> MotionResult:
        """Build a MotionResult from tile-grid detector outputs."""
        motion_field = self._build_motion_field(tile_direction, tile_confidence, height, width)
        saliency_map = self._expand_tile_grid(tile_confidence, height, width)
        magnitude_map = np.linalg.norm(motion_field, axis=2).astype(np.float32)
        angle_map = np.arctan2(motion_field[..., 1], motion_field[..., 0]).astype(np.float32)
        angle_map = np.where(magnitude_map > 0, angle_map, 0.0).astype(np.float32)

        active = saliency_map > 0
        if np.any(active):
            global_dx = float(np.mean(motion_field[..., 0][active]))
            global_dy = float(np.mean(motion_field[..., 1][active]))
        else:
            global_dx = 0.0
            global_dy = 0.0

        diagnostics = {
            'tile_size': self.tile_size,
            'frame_shape': (height, width),
            'total_tiles': int(tile_direction.size),
            'active_tiles': int(np.count_nonzero(tile_direction >= 0)),
            'motion_threshold': self.config.motion_threshold,
            'scales': self.config.scales,
        }
        if diagnostics_extra:
            diagnostics.update(diagnostics_extra)

        scene_quality = {
            'exposure_clip': float(np.mean((source_frame < 0.05) | (source_frame > 0.95))),
            'global_contrast': float(np.std(source_frame)),
        }

        ego_compensated_field = motion_field.copy()
        regions = self._extract_regions(saliency_map, ego_compensated_field)
        looming_score, ttc = self._compute_looming_and_ttc(
            ego_compensated_field, saliency_map, dt=self.config.default_dt
        )
        steering_signal = self._compute_steering_signal(saliency_map)

        return MotionResult(
            saliency_map=saliency_map,
            motion_field=motion_field,
            confidence_map=saliency_map.copy(),
            magnitude_map=magnitude_map,
            angle_map=angle_map,
            global_vector=(global_dx, global_dy),
            ego_motion_vector=None,
            ego_compensated_field=ego_compensated_field,
            scene_quality=scene_quality,
            regions=regions,
            looming_score=looming_score,
            time_to_contact=ttc,
            steering_signal=steering_signal,
            diagnostics=diagnostics,
            _direction_grids=direction_grids,
            _t4_grids=t4_grids,
            _t5_grids=t5_grids,
            _include_pathway=self.config.include_pathway_maps,
            _tile_direction=tile_direction,
            _tile_confidence=tile_confidence,
            _tile_size=self.tile_size,
            _frame_shape=(height, width),
        )

    def _detect_motion_grid_pair(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute a stateless vectorized motion grid from one frame pair.

        Uses rectified raw intensity as the correlator input (see
        :meth:`process_pair`). The stateful :meth:`update` path passes filtered
        contrast signals to the same grid correlator instead.
        """
        if self.config.flicker_suppression:
            # Remove the common-mode (global) brightness change by splitting the
            # median shift between both frames. A pure flash then yields identical
            # frames and therefore no spurious motion.
            global_shift = float(np.median(frame_curr - frame_prev))
            if global_shift != 0.0:
                frame_prev = np.clip(frame_prev + global_shift * 0.5, 0.0, 1.0)
                frame_curr = np.clip(frame_curr - global_shift * 0.5, 0.0, 1.0)

        on_prev = self._rectify_on(frame_prev).astype(np.float32)
        on_curr = self._rectify_on(frame_curr).astype(np.float32)
        off_prev = self._rectify_off(1.0 - frame_prev).astype(np.float32)
        off_curr = self._rectify_off(1.0 - frame_curr).astype(np.float32)

        return self._detect_motion_grid_stateful(on_prev, on_curr, off_prev, off_curr)

    def _detect_motion_grid_stateful(
        self,
        on_lowpass: np.ndarray,
        on_highpass: np.ndarray,
        off_lowpass: np.ndarray,
        off_highpass: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute multi-scale best direction and confidence for every tile in vectorized batches."""
        
        all_magnitudes: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        all_consistencies: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        all_t4: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        all_t5: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        
        for scale in self.config.scales:
            for direction in self.DIRECTIONS:
                mag, cons, t4_mag, t5_mag = self._correlate_direction_grid(
                    on_lowpass, on_highpass,
                    off_lowpass, off_highpass,
                    direction,
                    scale
                )
                all_magnitudes[direction].append(mag)
                all_consistencies[direction].append(cons)
                all_t4[direction].append(t4_mag)
                all_t5[direction].append(t5_mag)
                
        # Confidence-weighted fusion across scales
        magnitudes = {}
        consistencies = {}
        t4_fused = {}
        t5_fused = {}
        for direction in self.DIRECTIONS:
            # arrays of shape (scales, H, W)
            mags = np.stack(all_magnitudes[direction], axis=0)
            conss = np.stack(all_consistencies[direction], axis=0)
            t4s = np.stack(all_t4[direction], axis=0)
            t5s = np.stack(all_t5[direction], axis=0)
            
            # Weighted average of magnitudes by their confidence
            sum_conss = np.sum(conss, axis=0) + 1e-6
            magnitudes[direction] = np.sum(mags * conss, axis=0) / sum_conss
            consistencies[direction] = np.max(conss, axis=0)
            t4_fused[direction] = np.sum(t4s * conss, axis=0) / sum_conss
            t5_fused[direction] = np.sum(t5s * conss, axis=0) / sum_conss

        if self.config.opponent_subtraction:
            net_y = magnitudes['down'] - magnitudes['up']
            net_x = magnitudes['right'] - magnitudes['left']
            use_y = np.abs(net_y) > np.abs(net_x)
            
            tile_direction = np.full(net_x.shape, -1, dtype=np.int8)
            tile_direction = np.where(use_y & (net_y > 0), 1, tile_direction)
            tile_direction = np.where(use_y & (net_y <= 0), 0, tile_direction)
            tile_direction = np.where(~use_y & (net_x > 0), 3, tile_direction)
            tile_direction = np.where(~use_y & (net_x <= 0), 2, tile_direction)
            
            best_magnitude = np.where(use_y, np.abs(net_y), np.abs(net_x)).astype(np.float32)
        else:
            stack = np.stack([magnitudes['up'], magnitudes['down'], magnitudes['left'], magnitudes['right']], axis=0)
            best_magnitude = np.max(stack, axis=0)
            tile_direction = np.argmax(stack, axis=0).astype(np.int8)

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

        return tile_direction, tile_confidence, direction_grids, t4_fused, t5_fused
    def _correlate_direction_grid(
        self,
        on_delayed: np.ndarray,
        on_curr: np.ndarray,
        off_delayed: np.ndarray,
        off_curr: np.ndarray,
        direction: str,
        scale: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute motion magnitude and coherence for one direction over all tiles."""
        dy, dx = self.DIRECTION_OFFSETS[direction]
        dy, dx = dy * scale, dx * scale
        
        on_shifted = self._shift_no_wrap(on_curr, dy, dx)
        off_shifted = self._shift_no_wrap(off_curr, dy, dx)
        valid_shift = self._same_tile_shift_mask(on_curr.shape, dy, dx)

        on_motion = on_delayed * on_shifted * valid_shift
        off_motion = off_delayed * off_shifted * valid_shift

        t4_response, on_std = self._tile_mean_and_std(on_motion)
        t5_response, off_std = self._tile_mean_and_std(off_motion)
        t4_scale, t5_scale = self._pathway_scales(direction)

        total_scale = t4_scale + t5_scale + 1e-6
        scaled_t4 = t4_response * t4_scale
        scaled_t5 = t5_response * t5_scale
        magnitude = (
            (scaled_t4 + scaled_t5) / total_scale
        ).astype(np.float32)

        response_mean = t4_response + t5_response
        response_variability = on_std + off_std
        consistency = response_mean / (response_mean + response_variability + 1e-6)

        return magnitude, np.clip(consistency, 0.0, 1.0).astype(np.float32), scaled_t4, scaled_t5
    def _pathway_scales(self, direction: str) -> Tuple[float, float]:
        if direction in self._pathway_scale_cache:
            return self._pathway_scale_cache[direction]

        subtype = self.DIRECTION_SUBTYPES.get(direction, 'a')
        t4_weights_list = self.weights.get(f"t4_{subtype}") or self.weights.get(f"t4_{direction}")
        t5_weights_list = self.weights.get(f"t5_{subtype}") or self.weights.get(f"t5_{direction}")

        if t4_weights_list is not None and t5_weights_list is not None:
            if len(t4_weights_list) == len(self.inputs_order) and len(t5_weights_list) == len(self.inputs_order):
                t4_w = dict(zip(self.inputs_order, t4_weights_list, strict=False))
                t5_w = dict(zip(self.inputs_order, t5_weights_list, strict=False))
                
                t4_scale = t4_w.get('Mi1', 0) + t4_w.get('Mi4', 0) + t4_w.get('Mi9', 0) + t4_w.get('Tm3', 0)
                t5_scale = t5_w.get('Tm1', 0) + t5_w.get('Tm2', 0) + t5_w.get('Tm4', 0) + t5_w.get('Tm9', 0)
                
                scales = (float(t4_scale), float(t5_scale))
            else:
                scales = (float(sum(t4_weights_list)), float(sum(t5_weights_list)))
            self._pathway_scale_cache[direction] = scales
            return scales

        scales = (1.0, 1.0)
        self._pathway_scale_cache[direction] = scales
        return scales

    def _tile_mean_and_std(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-tile mean and std for a frame-sized 2D array."""
        H, W = values.shape[:2]
        tile_size = self.tile_size
        
        # cv2.resize with INTER_AREA acts as an exact block mean when downscaling by an integer factor
        new_w, new_h = W // tile_size, H // tile_size
        
        # We need exact integer multiples for cv2.INTER_AREA to act as perfect box filter.
        # It handles partial boundaries automatically, but we might just pad if they don't divide evenly.
        if H % tile_size != 0 or W % tile_size != 0:
            padded_h = ((H + tile_size - 1) // tile_size) * tile_size
            padded_w = ((W + tile_size - 1) // tile_size) * tile_size
            padded = np.zeros((padded_h, padded_w), dtype=np.float32)
            padded[:H, :W] = values
            new_w, new_h = padded_w // tile_size, padded_h // tile_size
            
            means = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sq_means = cv2.resize(padded * padded, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            means = cv2.resize(values, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sq_means = cv2.resize(values * values, (new_w, new_h), interpolation=cv2.INTER_AREA)

        variances = np.maximum(sq_means - means * means, 0.0)
        return means.astype(np.float32), np.sqrt(variances).astype(np.float32)

    def _same_tile_shift_mask(
        self,
        shape: Tuple[int, int],
        dy: int,
        dx: int
    ) -> np.ndarray:
        """Mask shifted pixels whose source stays inside the same detector tile."""
        height, width = shape
        cache_key = (height, width, dy, dx)
        cached = self._shift_mask_cache.get(cache_key)
        if cached is not None:
            self._shift_mask_cache.move_to_end(cache_key)
            return cached

        y_tiles = (np.arange(height) // self.tile_size)[:, None]
        x_tiles = (np.arange(width) // self.tile_size)[None, :]
        tile_ids = (y_tiles * ((width + self.tile_size - 1) // self.tile_size) + x_tiles + 1)
        shifted_tile_ids = self._shift_no_wrap(tile_ids.astype(np.int32), dy, dx)
        mask: np.ndarray = (shifted_tile_ids == tile_ids).astype(np.float32)
        self._shift_mask_cache[cache_key] = mask
        if len(self._shift_mask_cache) > self.SHIFT_MASK_CACHE_SIZE:
            self._shift_mask_cache.popitem(last=False)
        return mask

    def _expand_tile_grid(self, tile_grid: np.ndarray, height: int, width: int) -> np.ndarray:
        """Paint a tile grid back to frame resolution."""
        # cv2.resize with INTER_NEAREST is much faster than np.repeat
        return cv2.resize(tile_grid, (width, height), interpolation=cv2.INTER_NEAREST)

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


    def _extract_regions(self, saliency_map: np.ndarray, motion_field: np.ndarray) -> typing.List[Dict[str, Any]]:
        if not self.config.enable_regions:
            return []
            
        # Threshold saliency to find active regions
        active_mask = (saliency_map > 0.1).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(active_mask, connectivity=8)
        
        regions = []
        for i in range(1, num_labels):  # Skip 0 (background)
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 10:  # Ignore tiny noise regions
                continue
                
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]
            
            # Find dominant motion direction in this region
            region_mask = (labels == i)
            region_motion = motion_field[region_mask]
            mean_dy = float(np.mean(region_motion[:, 1]))
            mean_dx = float(np.mean(region_motion[:, 0]))
            
            regions.append({
                'bbox': (x, y, w, h),
                'centroid': (float(cx), float(cy)),
                'area': area,
                'dominant_vector': (mean_dx, mean_dy)
            })
            
        return regions

    def _compute_looming_and_ttc(
        self,
        motion_field: np.ndarray,
        saliency_map: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        """Estimate looming (expansion) magnitude and time-to-contact.

        Returns ``(looming_score, time_to_contact)``. ``time_to_contact`` is in
        seconds when a positive ``dt`` (frame interval) is supplied, otherwise in
        frames. It is ``None`` when expansion is below the confidence floor.
        """
        if not self.config.enable_looming:
            return 0.0, None
            
        H, W = motion_field.shape[:2]
        
        # We compute divergence: div(V) = d(Vx)/dx + d(Vy)/dy
        # Positive divergence indicates expansion (looming)
        dx = motion_field[..., 0]
        dy = motion_field[..., 1]
        
        # Central difference
        dVx_dx = np.zeros_like(dx)
        dVy_dy = np.zeros_like(dy)
        
        dVx_dx[:, 1:-1] = (dx[:, 2:] - dx[:, :-2]) / 2.0
        dVy_dy[1:-1, :] = (dy[2:, :] - dy[:-2, :]) / 2.0
        
        divergence = dVx_dx + dVy_dy
        
        # Only consider active regions
        active = saliency_map > 0.1
        if not np.any(active):
            return 0.0, None
            
        mean_div = float(np.mean(divergence[active]))
        
        looming_score = max(0.0, mean_div)
        
        ttc = None
        if looming_score > 0.05:
            # tau = 2 / divergence gives time in frame units; convert to seconds.
            ttc = 2.0 / looming_score
            if dt is not None and dt > 0:
                ttc *= dt
                
        return looming_score, ttc

    def _compute_steering_signal(self, saliency_map: np.ndarray) -> Dict[str, float]:
        if not self.config.enable_steering:
            return {'turn': 0.0, 'forward_caution': 0.0, 'collision_urgency': 0.0}
            
        H, W = saliency_map.shape[:2]
        mid = W // 2
        
        left_mass = np.sum(saliency_map[:, :mid])
        right_mass = np.sum(saliency_map[:, mid:])
        total_mass = left_mass + right_mass
        
        if total_mass < 1e-5:
            return {'turn': 0.0, 'forward_caution': 0.0, 'collision_urgency': 0.0}
            
        # Steer away from mass
        # turn > 0 is right, turn < 0 is left. 
        # If left mass is higher, we want to turn right (+).
        turn = (left_mass - right_mass) / total_mass
        
        forward_caution = min(1.0, total_mass / (H * W * 0.1)) # if 10% of screen is active, max caution
        collision_urgency = min(1.0, total_mass / (H * W * 0.3))
        
        return {
            'turn': float(turn),
            'forward_caution': float(forward_caution),
            'collision_urgency': float(collision_urgency)
        }

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB/BGR to grayscale."""
        if frame.ndim == 3:
            if frame.shape[2] < 3:
                raise ValueError("color frames must have at least 3 channels")
            if self.config.input_color not in ("rgb", "bgr"):
                raise ValueError("input_color must be either 'rgb' or 'bgr'")
            conversion = cv2.COLOR_RGB2GRAY
            if self.config.input_color == "bgr":
                conversion = cv2.COLOR_BGR2GRAY
            res = cv2.cvtColor(frame[..., :3], conversion)
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
        return typing.cast(np.ndarray, np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False))

    @staticmethod
    def _shift_no_wrap(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Shift a 2D array with zero padding instead of toroidal wraparound."""
        shifted = np.empty_like(frame)
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
            shifted.fill(0)
            return shifted

        # Zero out the regions that are left empty by the shift
        if dst_y0 > 0:
            shifted[:dst_y0, :] = 0
        if dst_y1 < height:
            shifted[dst_y1:, :] = 0
        if dst_x0 > 0:
            shifted[dst_y0:dst_y1, :dst_x0] = 0
        if dst_x1 < width:
            shifted[dst_y0:dst_y1, dst_x1:] = 0

        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
        return shifted


if __name__ == "__main__":
    # Test detector initialization
    detector = MotionDetector(tile_size=8)
    print("MotionDetector ready")
