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
    scales: Tuple[int, ...] = (1, 2, 4, 8)
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
    
    # Robotics outputs. Disabled by default: looming/TTC and steering are
    # research-grade heuristics, not validated safety systems. Enable them
    # explicitly (or use ``preset_looming``) when you have validated the
    # behaviour for your scene and sensor.
    enable_regions: bool = False
    enable_looming: bool = False
    enable_steering: bool = False
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
    def preset_education(cls, **kwargs: Any) -> "NeuroSightConfig":
        """Slower, highly explainable settings with default robustness disabled."""
        defaults: Dict[str, Any] = dict(
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
    def preset_edge_fast(cls, **kwargs: Any) -> "NeuroSightConfig":
        """Minimal latency settings optimized for Raspberry Pi, turning off some heavy robotics processing."""
        defaults: Dict[str, Any] = dict(
            scales=(1,),
            blur_handling=False,
            temporal_denoising=False,
            ego_motion_compensation=False,
            enable_looming=False,
            enable_regions=False,
            enable_steering=False,
            include_pathway_maps=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def preset_robust(cls, **kwargs: Any) -> "NeuroSightConfig":
        """High noise suppression for complex lighting or bumpy movement."""
        defaults: Dict[str, Any] = dict(
            temporal_denoising=True,
            denoising_alpha=0.3,
            flicker_suppression=True,
            ego_motion_compensation=True,
            blur_handling=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def preset_looming(cls, **kwargs: Any) -> "NeuroSightConfig":
        """Settings for expansion/TTC experiments (research, not safety-rated)."""
        defaults: Dict[str, Any] = dict(
            ego_motion_compensation=True,
            enable_looming=True,
            enable_steering=True,
            enable_regions=True,
            scales=(1, 2, 4, 8),
        )
        defaults.update(kwargs)
        return cls(**defaults)

@dataclass(eq=False)
class MotionResult:
    """Structured result returned by the newer motion API.

    The frame-resolution direction and pathway maps are expanded lazily from
    their tile-resolution grids, so constructing a result is cheap and those
    arrays are only allocated when a caller actually reads them.

    Some arrays are aliases to avoid full-frame copies: ``confidence_map`` is
    the same array as ``saliency_map``, and ``ego_compensated_field`` aliases
    ``motion_field`` when ego-motion compensation is disabled. Treat all
    result arrays as read-only.
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

    # Bounds for cached per-shape shift masks. Boolean masks are cheap, but a
    # 4K frame still costs ~8 MB per scale/direction, so cap total bytes too.
    SHIFT_MASK_CACHE_SIZE = 64
    SHIFT_MASK_CACHE_BYTES = 32 * 1024 * 1024

    # Stateless pair path: per-tile temporal change needed before correlation
    # output is trusted, and the half-saturation scale for the soft gate.
    PAIR_ACTIVITY_FLOOR = 1e-3
    PAIR_ACTIVITY_SCALE = 2e-2

    def __init__(
        self,
        connectome_weights_path: Optional[str] = None,
        tile_size: int = 8,
        config: Optional[NeuroSightConfig] = None,
    ):
        """
        Initialize motion detector with biological weights.
        
        Args:
            connectome_weights_path: Path to connectome_weights.json (exported from Phase 1).
                If None, attempts to load from package data.
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
        self._shift_mask_cache_bytes = 0
        self._pathway_scale_cache: Dict[str, Tuple[float, float]] = {}
        
        self.sigmoid_gain = self.DEFAULT_SIGMOID_GAIN
        self.sigmoid_threshold = self.DEFAULT_SIGMOID_THRESHOLD
        
        self.reset()
        logger.info("MotionDetector initialized (tile_size=%s)", self.tile_size)

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
        return self._validate_weights(WEIGHTS, "bundled weights")

    @classmethod
    def _validate_weights(cls, data: Any, source: str) -> dict:
        """Validate the structure and schema of a connectome weights payload.

        A valid payload must define either the four subtype vectors
        (``t4_a``..``t4_d`` / ``t5_a``..``t5_d``) or the four direction-named
        vectors (``t4_up``..).  When ``inputs_order`` is present every vector
        must match its length; without it, all vectors must still agree on a
        single length so that ``_pathway_scales`` cannot silently fall back to
        uniform gains.
        """
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

        inputs_order = data.get('inputs_order')
        expected_length: Optional[int] = None
        if inputs_order is not None:
            if (
                not isinstance(inputs_order, (list, tuple))
                or not inputs_order
                or any(not isinstance(name, str) or not name for name in inputs_order)
            ):
                raise ValueError(
                    f"'inputs_order' from {source} must be a non-empty list of names."
                )
            if len(set(inputs_order)) != len(inputs_order):
                raise ValueError(
                    f"'inputs_order' from {source} contains duplicate names."
                )
            expected_length = len(inputs_order)

        lengths: set[int] = set()
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
            lengths.add(int(arr.size))

        subtype_keys = {f"t4_{name}" for name in "abcd"} | {
            f"t5_{name}" for name in "abcd"
        }
        direction_keys = {
            f"t4_{name}" for name in ("up", "down", "left", "right")
        } | {f"t5_{name}" for name in ("up", "down", "left", "right")}
        present = set(weights)
        if not subtype_keys <= present and not direction_keys <= present:
            missing = sorted((subtype_keys if not subtype_keys <= present else direction_keys) - present)
            raise ValueError(
                f"Connectome weights from {source} must define t4/t5 vectors for "
                f"all four directions (subtypes a-d or up/down/left/right); "
                f"missing {', '.join(missing)}."
            )

        if expected_length is not None and lengths != {expected_length}:
            raise ValueError(
                f"Connectome weights from {source} must all have length "
                f"{expected_length} to match 'inputs_order'; got lengths {sorted(lengths)}."
            )
        if expected_length is None and len(lengths) != 1:
            raise ValueError(
                f"Connectome weights from {source} must all have the same length; "
                f"got lengths {sorted(lengths)}."
            )

        return data

    def _load_weights(self, filepath: Optional[str]) -> dict:
        """Load and validate weights; use the bundled data when no path is given."""
        if not filepath:
            return self._fallback_weights()

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            # An explicitly supplied path is a user contract. Silently falling
            # back to different weights hides typos and changes model output.
            raise FileNotFoundError(
                f"Connectome weights file '{filepath}' does not exist. "
                "Pass None (or omit the argument) to use the bundled weights."
            ) from exc
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

    def _pair_rectify_on(self, intensity: np.ndarray) -> np.ndarray:
        """ON rectifier with fixed default parameters, independent of stream state."""
        return (intensity / (self.DEFAULT_SIGMOID_THRESHOLD + intensity + 1e-6)) * self.DEFAULT_SIGMOID_GAIN

    def _pair_rectify_off(self, intensity: np.ndarray) -> np.ndarray:
        """OFF rectifier with fixed default parameters, independent of stream state."""
        return self._pair_rectify_on(intensity)

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
        # Temporal-contrast delay line. Correlating contrast (rather than raw
        # intensity) removes static structure, so a still scene contributes no
        # motion energy even when the correlator is evaluated at nonzero shifts.
        self._contrast_lowpass: Optional[np.ndarray] = None
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

        The first frame seeds the temporal-contrast delay line and returns
        ``None``. Subsequent calls correlate the delayed contrast (temporal
        derivative, low-passed with ``tau_lowpass``) against the current
        contrast at the configured spatial scales. A static scene therefore
        produces no motion energy, independent of spatial structure.

        If the resolution changes, state resets and the first frame of the new
        resolution returns ``None``.
        """
        if frame_curr is None:
            raise ValueError("frame_curr must be a numpy array")

        curr = np.asarray(frame_curr)
        if curr.ndim not in (2, 3) or curr.size == 0:
            raise ValueError(
                f"frame_curr must be a non-empty 2D or 3D array, got shape {curr.shape}"
            )
        if not np.all(np.isfinite(curr)):
            # A NaN here would propagate into the adaptive threshold and disable
            # detection until reset(), so reject it before any state is touched.
            raise ValueError("frame_curr contains NaN or Inf values")
        if curr.ndim == 3:
            curr = self._to_grayscale(curr)

        curr = self._normalize_frame(curr)

        H, W = curr.shape[:2]

        # Check the resolution before touching temporal-denoising state. A
        # size change invalidates the running average, and blending frames of
        # different shapes would raise instead of resetting cleanly.
        if self._last_frame_shape is not None and self._last_frame_shape != (H, W):
            logger.info("Resolution changed, resetting temporal state.")
            self.reset()

        self._last_frame_shape = (H, W)

        if self.config.blur_handling:
            curr = cv2.GaussianBlur(curr, (3, 3), 0).astype(np.float32)

        if self.config.temporal_denoising:
            if self._smoothed_frame is None:
                self._smoothed_frame = curr.copy()
            else:
                self._smoothed_frame = (
                    self.config.denoising_alpha * curr
                    + (1.0 - self.config.denoising_alpha) * self._smoothed_frame
                )
            process_frame = self._smoothed_frame
        else:
            process_frame = curr

        # Frame interval: prefer the caller's timestamps, then the default.
        if timestamp is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt <= 0:
                dt = self.config.default_dt
        else:
            dt = self.config.default_dt
        self._last_timestamp = timestamp

        if self._last_frame is None:
            # Seed the delay line with a zero contrast frame. The first
            # inter-frame difference is not available until the next call, and
            # correlating the current frame with itself would be even in
            # direction (it cannot distinguish left from right).
            self._contrast_lowpass = np.zeros_like(process_frame, dtype=np.float32)
            self._last_frame = process_frame.copy()
            return None

        # Compare the current frame against the previous *processed* frame.
        # With temporal denoising enabled this is a high-pass relative to a
        # running average, which preserves sensitivity to new motion instead of
        # differencing two already-smoothed frames (which halves the contrast).
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
            # Smooth the tile-mean estimate back to frame resolution. Nearest-
            # neighbour expansion leaves 8-pixel seams that correlate with
            # themselves at larger shifts and fabricate vertical/horizontal
            # motion from an otherwise one-directional stimulus.
            expanded_mean = self._expand_tile_grid_smooth(local_mean, H, W)
            delta = delta - expanded_mean

        self._last_frame = process_frame.copy()

        mean_abs_contrast = float(np.mean(np.abs(delta)))
        self.sigmoid_threshold = (
            (1.0 - self.config.adaptation_rate) * self.sigmoid_threshold
            + self.config.adaptation_rate * mean_abs_contrast
        )
        self.sigmoid_threshold = float(np.clip(self.sigmoid_threshold, 0.05, 0.5))

        # Reichardt delay line: correlate the previous temporal contrast with
        # the current temporal contrast. Both signals are zero in a static
        # scene, and their spatial offset encodes the direction of motion.
        delayed = self._contrast_lowpass
        if delayed is None or delayed.shape != delta.shape:
            delayed = np.zeros_like(delta, dtype=np.float32)

        on_delayed = self._rectify_on(np.clip(delayed, 0.0, None)).astype(np.float32)
        off_delayed = self._rectify_off(np.clip(-delayed, 0.0, None)).astype(np.float32)
        on_curr = self._rectify_on(np.clip(delta, 0.0, None)).astype(np.float32)
        off_curr = self._rectify_off(np.clip(-delta, 0.0, None)).astype(np.float32)

        # Low-pass the delay-line signal. tau_lowpass controls how quickly the
        # previous contrast is forgotten; at the default dt this is a mild
        # smoothing that leaves one-frame displacement cues intact.
        alpha_lp = 1.0 - np.exp(-dt / max(self.config.tau_lowpass, 1e-6))
        alpha_lp = float(np.clip(alpha_lp, 0.0, 1.0))
        self._contrast_lowpass = (
            (1.0 - alpha_lp) * delayed + alpha_lp * delta
        ).astype(np.float32)

        tile_direction, tile_confidence, direction_grids, t4_grids, t5_grids = self._detect_motion_grid_stateful(
            on_delayed, on_curr, off_delayed, off_curr
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
        if self.config.ego_motion_compensation:
            ego_compensated_field = motion_field.copy()
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
        else:
            # Read-only alias: avoid a full-frame copy when compensation is off.
            ego_compensated_field = motion_field
            
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
            # Saliency is the per-tile confidence painted back to frame
            # resolution; expose the same array read-only instead of duplicating it.
            confidence_map=saliency_map,
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
        *spatial high-pass* (local contrast) of the two frames. A two-frame
        input has no temporal delay line, so high-pass features remove the
        static background that would otherwise dominate. A per-tile temporal
        change gate suppresses residual correlation on still scenes. The
        stateful :meth:`update` path instead correlates the previous *temporal
        contrast* against the current temporal contrast. Both paths estimate
        direction, but their magnitudes/confidences are not numerically
        identical: the pair path has no delay-line warm-up.

        Streaming-only options are intentionally not applied here:
        ``contrast_normalization``, ``lateral_inhibition``, and
        ``ego_motion_compensation`` require history and are ignored by this
        stateless path. ``flicker_suppression`` and ``blur_handling`` are
        applied to the input pair.

        Use this when you have exactly two frames and want an immediate result;
        use :meth:`update` for continuous streams.
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

        # The stateless pair path has no ego-motion history, so expose a
        # read-only alias rather than copying a full-frame field.
        ego_compensated_field = motion_field
        regions = self._extract_regions(saliency_map, ego_compensated_field)
        looming_score, ttc = self._compute_looming_and_ttc(
            ego_compensated_field, saliency_map, dt=self.config.default_dt
        )
        steering_signal = self._compute_steering_signal(saliency_map)

        return MotionResult(
            saliency_map=saliency_map,
            motion_field=motion_field,
            # Saliency is the per-tile confidence painted back to frame
            # resolution; expose the same array read-only instead of duplicating it.
            confidence_map=saliency_map,
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

        The pair path has no temporal history, so it uses a spatial high-pass
        (local contrast) representation instead of the streaming path's
        temporal contrast delay line. High-pass features remove the static DC
        component that otherwise dominates the correlator at nonzero shifts.
        A per-tile temporal-change gate then suppresses residual response on
        still scenes and global brightness changes.
        """
        if self.config.flicker_suppression:
            # Remove the common-mode (global) brightness change by splitting the
            # median shift between both frames. A pure flash then yields identical
            # frames and therefore no spurious motion.
            global_shift = float(np.median(frame_curr - frame_prev))
            if global_shift != 0.0:
                frame_prev = np.clip(frame_prev + global_shift * 0.5, 0.0, 1.0)
                frame_curr = np.clip(frame_curr - global_shift * 0.5, 0.0, 1.0)

        change = (frame_curr - frame_prev).astype(np.float32)

        # A local contrast operator scaled to the tile size keeps edges and
        # polarity while discarding the unchanging background. Use the tile
        # size itself rather than half of it so broad, slowly varying patterns
        # (periods up to ~64 px at the default tile size) survive the high-pass
        # instead of vanishing before correlation.
        sigma = float(np.clip(self.tile_size, 1.0, 12.0))
        prev_hp = frame_prev - cv2.GaussianBlur(frame_prev, (0, 0), sigma)
        curr_hp = frame_curr - cv2.GaussianBlur(frame_curr, (0, 0), sigma)

        on_prev = self._pair_rectify_on(np.clip(prev_hp, 0.0, None)).astype(np.float32)
        off_prev = self._pair_rectify_off(np.clip(-prev_hp, 0.0, None)).astype(np.float32)
        on_curr = self._pair_rectify_on(np.clip(curr_hp, 0.0, None)).astype(np.float32)
        off_curr = self._pair_rectify_off(np.clip(-curr_hp, 0.0, None)).astype(np.float32)

        tile_direction, tile_confidence, direction_grids, t4_grids, t5_grids = (
            self._detect_motion_grid_stateful(on_prev, on_curr, off_prev, off_curr)
        )

        # Gate correlation confidence by how much the input actually changed.
        # Without this, static texture self-correlates at every shift and can
        # produce low-level directional output.
        change_grid, _ = self._tile_mean_and_std(np.abs(change))
        gate = change_grid / (change_grid + self.PAIR_ACTIVITY_SCALE)
        gate = np.where(change_grid >= self.PAIR_ACTIVITY_FLOOR, gate, 0.0).astype(np.float32)

        tile_confidence = (tile_confidence * gate).astype(np.float32)
        tile_direction = np.where(tile_confidence > 0.0, tile_direction, -1).astype(np.int8)
        direction_grids = {key: (grid * gate).astype(np.float32) for key, grid in direction_grids.items()}
        t4_grids = {key: (grid * gate).astype(np.float32) for key, grid in t4_grids.items()}
        t5_grids = {key: (grid * gate).astype(np.float32) for key, grid in t5_grids.items()}

        # Rectified onset/offset transients are part of the T4/T5 channel
        # response. Motion correlation supplies direction, while the transient
        # term preserves ON/OFF selectivity for objects that appear or vanish
        # without correlated displacement (the previous frame has no local
        # contrast at that location). Distribute it across directions because
        # a pure onset has no defined direction.
        on_transient, _ = self._tile_mean_and_std(
            self._pair_rectify_on(np.clip(change, 0.0, None))
        )
        off_transient, _ = self._tile_mean_and_std(
            self._pair_rectify_off(np.clip(-change, 0.0, None))
        )
        share = 1.0 / len(self.DIRECTIONS)
        t4_grids = {
            key: (grid + on_transient * share).astype(np.float32)
            for key, grid in t4_grids.items()
        }
        t5_grids = {
            key: (grid + off_transient * share).astype(np.float32)
            for key, grid in t5_grids.items()
        }

        return tile_direction, tile_confidence, direction_grids, t4_grids, t5_grids

    def _detect_motion_grid_stateful(
        self,
        on_delayed: np.ndarray,
        on_curr: np.ndarray,
        off_delayed: np.ndarray,
        off_curr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute multi-scale best direction and confidence for every tile in vectorized batches."""
        
        all_magnitudes: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        all_consistencies: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        all_t4: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        all_t5: Dict[str, List[np.ndarray]] = {direction: [] for direction in self.DIRECTIONS}
        
        # Ascending order makes "first scale within 90% of the best score"
        # mean "smallest scale", independent of how the user ordered the tuple.
        for scale in sorted(self.config.scales):
            for direction in self.DIRECTIONS:
                mag, cons, t4_mag, t5_mag = self._correlate_direction_grid(
                    on_delayed, on_curr,
                    off_delayed, off_curr,
                    direction,
                    scale
                )
                all_magnitudes[direction].append(mag)
                all_consistencies[direction].append(cons)
                all_t4[direction].append(t4_mag)
                all_t5[direction].append(t5_mag)
                
        # Scale selection: keep the response of the scale that best explains
        # each tile, rather than averaging all scales together.
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
            
            # Select the scale that best explains each tile instead of averaging
            # all scales together. A confidence-weighted average lets a large
            # scale with a weak, ambiguous response dilute a strong small-scale
            # response at low speeds (and vice versa), which made the reported
            # direction depend on sub-tile phase. The score combines response
            # strength with reliability so a clean small-scale peak wins at low
            # speed while a large scale wins only when the displacement needs it.
            scores = mags * conss
            # Prefer the smallest scale that is nearly as good as the best one.
            # Coarse scales are only selected when they explain the displacement
            # materially better, which keeps low-speed responses from being
            # diluted by an ambiguous large-scale correlation.
            best_score = np.max(scores, axis=0, keepdims=True)
            eligible = scores >= 0.9 * best_score
            best_scale = np.argmax(eligible, axis=0)[None, ...]
            magnitudes[direction] = np.take_along_axis(mags, best_scale, axis=0)[0]
            consistencies[direction] = np.take_along_axis(conss, best_scale, axis=0)[0]
            t4_fused[direction] = np.take_along_axis(t4s, best_scale, axis=0)[0]
            t5_fused[direction] = np.take_along_axis(t5s, best_scale, axis=0)[0]

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
        """Correlate delayed and current signals for one direction over all tiles.

        ``on_delayed``/``off_delayed`` are the delayed (or previous) rectified
        ON/OFF signals and ``on_curr``/``off_curr`` the current ones. The
        correlator is evaluated over every pixel whose shifted sample stays
        inside the frame; pooling into tiles happens afterwards, so
        displacements that match the tile size are not erased.
        """
        dy, dx = self.DIRECTION_OFFSETS[direction]
        dy, dx = dy * scale, dx * scale
        
        on_shifted = self._shift_no_wrap(on_curr, dy, dx)
        off_shifted = self._shift_no_wrap(off_curr, dy, dx)
        # Mask only pixels that leave the frame. Restricting the correlation to
        # same-tile pixels creates periodic blind spots at tile-aligned
        # displacements and makes the result depend on sub-tile phase.
        valid_shift = self._shift_validity_mask(on_curr.shape, dy, dx)

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

    def _shift_validity_mask(
        self,
        shape: Tuple[int, int],
        dy: int,
        dx: int
    ) -> np.ndarray:
        """Boolean mask of pixels whose shifted source stays inside the frame.

        Correlating over the whole valid frame (and pooling the result into
        tiles afterwards) preserves every displacement up to the frame border.
        Masking to *tile-local* pixels, as the previous implementation did,
        erased motion whose displacement matched the tile size.
        """
        height, width = shape
        cache_key = (height, width, dy, dx)
        cached = self._shift_mask_cache.get(cache_key)
        if cached is not None:
            self._shift_mask_cache.move_to_end(cache_key)
            return cached

        mask = np.ones((height, width), dtype=bool)
        if dy > 0:
            mask[:dy, :] = False
        elif dy < 0:
            mask[dy:, :] = False
        if dx > 0:
            mask[:, :dx] = False
        elif dx < 0:
            mask[:, dx:] = False

        if abs(dy) >= height or abs(dx) >= width:
            mask.fill(False)

        self._shift_mask_cache[cache_key] = mask
        self._shift_mask_cache_bytes += mask.nbytes
        while (
            len(self._shift_mask_cache) > self.SHIFT_MASK_CACHE_SIZE
            or self._shift_mask_cache_bytes > self.SHIFT_MASK_CACHE_BYTES
        ):
            _, evicted = self._shift_mask_cache.popitem(last=False)
            self._shift_mask_cache_bytes -= evicted.nbytes
        return mask

    def _expand_tile_grid(self, tile_grid: np.ndarray, height: int, width: int) -> np.ndarray:
        """Paint a tile grid back to frame resolution."""
        # cv2.resize with INTER_NEAREST is much faster than np.repeat
        return cv2.resize(tile_grid, (width, height), interpolation=cv2.INTER_NEAREST)

    def _expand_tile_grid_smooth(
        self, tile_grid: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Smoothly upsample a tile grid for continuous fields (not masks)."""
        return cv2.resize(
            tile_grid, (width, height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

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
        if prev.size == 0 or curr.size == 0:
            raise ValueError("frames must be non-empty")
        if not (np.all(np.isfinite(prev)) and np.all(np.isfinite(curr))):
            raise ValueError("frames contain NaN or Inf values")

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
        """Estimate looming (radial expansion) and a heuristic time-to-contact.

        The old implementation differentiated a winner-take-all motion field.
        For a symmetric expanding object the opponent stage cancels opposite
        directions before divergence is measured, so the score stayed near zero.
        This version measures how strongly the surviving motion vectors point
        *away from* the active region's centroid, then suppresses the score when
        the field is coherent in one direction (translation, not expansion).

        Returns ``(looming_score, time_to_contact)``. ``looming_score`` is a
        dimensionless expansion evidence score in [0, 1]. ``time_to_contact`` is
        a heuristic: it assumes the confidence-weighted field is proportional to
        pixel displacement, using the mean configured scale as the conversion.
        It is in seconds when a positive ``dt`` is supplied, otherwise in
        frames, and is ``None`` below the expansion threshold. It is **not** a
        calibrated metric TTC and must not be used as a safety signal.
        """
        if not self.config.enable_looming:
            return 0.0, None

        active = saliency_map > 0.1
        if int(np.count_nonzero(active)) < 4:
            return 0.0, None

        ys, xs = np.nonzero(active)
        weights = saliency_map[ys, xs].astype(np.float64)
        vx = motion_field[ys, xs, 0].astype(np.float64)
        vy = motion_field[ys, xs, 1].astype(np.float64)

        total_weight = float(np.sum(weights))
        if total_weight <= 1e-9:
            return 0.0, None

        cx = float(np.average(xs, weights=weights))
        cy = float(np.average(ys, weights=weights))

        dx = xs.astype(np.float64) - cx
        dy = ys.astype(np.float64) - cy
        radius = np.hypot(dx, dy)
        valid = radius > 1.0
        if not np.any(valid):
            return 0.0, None

        weights = weights[valid]
        vx = vx[valid]
        vy = vy[valid]
        radius = radius[valid]
        ux = dx[valid] / radius
        uy = dy[valid] / radius

        speed = np.hypot(vx, vy)
        mean_speed = float(np.average(speed, weights=weights))
        if mean_speed <= 1e-9:
            return 0.0, None

        radial_speed = vx * ux + vy * uy
        outflow = float(np.average(radial_speed, weights=weights) / mean_speed)

        # A purely translating field has a large mean vector; radial flow has a
        # near-zero mean. Suppress translation so it cannot masquerade as looming.
        mean_vx = float(np.average(vx, weights=weights))
        mean_vy = float(np.average(vy, weights=weights))
        coherence = float(np.hypot(mean_vx, mean_vy) / mean_speed)
        loom = max(0.0, outflow) * max(0.0, 1.0 - min(coherence, 1.0))
        looming_score = float(np.clip(loom, 0.0, 1.0))

        ttc = None
        if looming_score > 0.05:
            mean_radius = float(np.average(radius, weights=weights))
            mean_radial_speed = float(np.average(radial_speed, weights=weights))
            # Convert confidence to pixels/frame with the mean correlation scale.
            pixel_scale = float(np.mean(self.config.scales)) if self.config.scales else 1.0
            denominator = mean_radial_speed * max(pixel_scale, 1e-6)
            if denominator > 1e-9:
                ttc = mean_radius / denominator
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
        if frame.size == 0:
            raise ValueError("frame must be non-empty")
        if not np.all(np.isfinite(frame)):
            raise ValueError("frame contains NaN or Inf values")
        if np.issubdtype(frame.dtype, np.integer):
            info = np.iinfo(frame.dtype)
            normalized = frame.astype(np.float32) / float(info.max)
        else:
            normalized = frame.astype(np.float32, copy=False)
            if normalized.size and float(np.max(normalized)) > 1.0:
                if float(np.max(normalized)) > 255.0:
                    raise ValueError(
                        "float frames must be in [0, 1] or [0, 255]; got values above 255"
                    )
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
