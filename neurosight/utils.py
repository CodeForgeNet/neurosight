"""
Phase 3: Frame Pipeline & Utilities
Frame buffering, video I/O, normalization, performance metrics.
"""

from typing import Optional, Tuple, Iterator, Any
import typing
from collections import deque
import numpy as np
import cv2
import logging
import time

logger = logging.getLogger(__name__)


class FrameBuffer:
    """
    Temporal buffer for consecutive frames.
    Maintains frame N-1 and frame N for correlator.
    """

    def __init__(self, buffer_size: int = 2):
        """
        Initialize frame buffer.
        
        Args:
            buffer_size: Number of frames to keep in history (default: 2 for prev/curr)
        """
        self.buffer: deque = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def push(self, frame: np.ndarray) -> bool:
        """
        Push a new frame into buffer.
        
        Args:
            frame: Frame array (H, W) or (H, W, 3)
        
        Returns:
            True if buffer is full (ready for processing)
        """
        self.buffer.append(frame.copy())
        return len(self.buffer) == self.buffer_size

    def get_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get (previous_frame, current_frame) pair.
        
        Returns:
            (frame_N-1, frame_N) if buffer is full, else None
        """
        if len(self.buffer) == self.buffer_size:
            frames = list(self.buffer)
            return frames[0], frames[1]
        return None

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for processing."""
        return len(self.buffer) == self.buffer_size

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()


class VideoProcessor:
    """
    Video I/O and frame processing pipeline.
    Handles reading, resizing, normalization.
    """

    def __init__(self, resize_width: Optional[int] = 320):
        """
        Initialize video processor.
        
        Args:
            resize_width: Resize frames to this width for efficiency (aspect ratio preserved)
                         None = no resizing
        """
        self.resize_width = resize_width

    def read_video(self, video_path: str) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Read video file frame-by-frame.
        
        Args:
            video_path: Path to video file
        
        Yields:
            (frame_index, frame_array)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise IOError(f"Cannot read video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {total_frames} frames @ {fps} FPS")
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if requested
                if self.resize_width is not None:
                    frame = self._resize_frame(frame, self.resize_width)
                
                yield frame_idx, frame
                frame_idx += 1
                
        finally:
            cap.release()

    def _resize_frame(self, frame: np.ndarray, width: int) -> np.ndarray:
        """
        Resize frame preserving aspect ratio.
        
        Args:
            frame: Input frame
            width: Target width
        
        Returns:
            Resized frame
        """
        H, W = frame.shape[:2]
        scale = width / W
        new_H = int(H * scale)
        return cv2.resize(frame, (width, new_H), interpolation=cv2.INTER_LINEAR)

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB frame to grayscale."""
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def normalize_intensity(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize intensity to [0, 1].
        
        Args:
            frame: Frame in [0, 255]
        
        Returns:
            Frame in [0, 1]
        """
        return frame.astype(np.float32) / 255.0


class PerformanceMonitor:
    """
    Track performance metrics: FPS, latency, memory.
    """

    def __init__(self):
        self.frame_times = deque(maxlen=30)  # Last 30 frames
        self.start_time = time.time()
        self.frame_count = 0

    def record_frame(self, processing_time: float):
        """
        Record processing time for a frame.
        
        Args:
            processing_time: Time in seconds to process frame
        """
        self.frame_times.append(processing_time)
        self.frame_count += 1

    def get_fps(self) -> float:
        """Get average FPS over last 30 frames."""
        if len(self.frame_times) == 0:
            return 0.0
        avg_time = np.mean(list(self.frame_times))
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_latency_ms(self) -> float:
        """Get average frame processing latency in milliseconds."""
        if len(self.frame_times) == 0:
            return 0.0
        return np.mean(list(self.frame_times)) * 1000.0

    def report(self) -> str:
        """Generate performance report."""
        fps = self.get_fps()
        latency = self.get_latency_ms()
        elapsed = time.time() - self.start_time
        
        return (
            f"Frames: {self.frame_count} | "
            f"FPS: {fps:.2f} | "
            f"Latency: {latency:.2f}ms | "
            f"Elapsed: {elapsed:.1f}s"
        )


class SaliencyMapVisualizer:
    """
    Convert saliency map and motion vectors to visualization.
    """

    @staticmethod
    def apply_colormap(saliency: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Apply colormap to saliency heatmap.
        
        Args:
            saliency: Grayscale saliency map [0, 1]
            colormap: OpenCV colormap (JET, VIRIDIS, etc.)
        
        Returns:
            RGB image (H, W, 3)
        """
        # Scale to [0, 255]
        saliency_8bit = (np.clip(saliency, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(saliency_8bit, colormap)
        
        # Convert BGR to RGB
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    @staticmethod
    def overlay_vectors(
        frame: np.ndarray,
        motion_vectors: dict,
        tile_size: int = 8,
        arrow_scale: float = 1.0
    ) -> np.ndarray:
        """
        Overlay motion vectors on frame.
        
        Args:
            frame: Input frame (H, W, 3) in RGB
            motion_vectors: Dict from detector.process_frame()
            tile_size: Size of tiles
            arrow_scale: Scale of arrow rendering
        
        Returns:
            Frame with arrows overlaid
        """
        output = frame.copy()
        
        # Direction to arrow offset mapping (in pixels)
        arrow_offsets = {
            'up': (0, -arrow_scale * tile_size / 3),
            'down': (0, arrow_scale * tile_size / 3),
            'left': (-arrow_scale * tile_size / 3, 0),
            'right': (arrow_scale * tile_size / 3, 0)
        }
        
        for tile_key, data in motion_vectors.items():
            direction = data['direction']
            confidence = data['confidence']
            y, x = data['y'], data['x']
            
            if direction is None or confidence < 0.1:
                continue
            
            # Tile center
            cy = int(y + tile_size / 2)
            cx = int(x + tile_size / 2)
            
            # Arrow endpoint
            dx, dy = arrow_offsets[direction]
            ex, ey = int(cx + dx), int(cy + dy)
            
            # Color intensity by confidence (RGB: Red for high confidence, Blue for low)
            color_intensity = int(255 * confidence)
            color = (color_intensity, 100, 255 - color_intensity)  # RGB
            
            # Draw arrow
            cv2.arrowedLine(
                output,
                (cx, cy), (ex, ey),
                color,
                thickness=2,
                tipLength=0.3
            )
        
        return typing.cast(np.ndarray, output)


class OutputWriter:
    """
    Write results to video file or images.
    """

    def __init__(self, output_path: str, fps: float = 30.0, frame_size: Optional[Tuple[int, int]] = None):
        """
        Initialize output writer.
        
        Args:
            output_path: Path to output video file (.mp4, .avi)
            fps: Frames per second
            frame_size: Frame size (width, height)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None

    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to output video.
        
        Args:
            frame: RGB frame (H, W, 3)
        """
        if self.writer is None:
            if self.frame_size is None:
                self.frame_size = (frame.shape[1], frame.shape[0])
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )
            
            if not self.writer.isOpened():
                raise IOError(f"Failed to open output writer: {self.output_path}")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def release(self):
        """Close output writer."""
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Output saved: {self.output_path}")


if __name__ == "__main__":
    # Test utilities
    buffer = FrameBuffer()
    processor = VideoProcessor(resize_width=320)
    monitor = PerformanceMonitor()
    
    print("Pipeline utilities ready")
