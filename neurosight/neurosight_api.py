"""
High-level NeuroSight API.

User-friendly interface: process a video or frame pair in a few lines.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from .detector import MotionDetector, MotionResult, NeuroSightConfig
from .utils import OutputWriter, PerformanceMonitor, SaliencyMapVisualizer, VideoProcessor

logger = logging.getLogger(__name__)


class NeuroSightDetector:
    """
    User-facing API for NeuroSight motion detection.
    
    Example:
        detector = NeuroSightDetector()
        aggregate_saliency, _ = detector.process_video("input.mp4", output_video="output.mp4")
        # Pass return_vectors=True to also get per-frame tile dictionaries.
    """

    def __init__(
        self,
        connectome_path: Optional[str] = None,
        tile_size: int = 8,
        resize_width: Optional[int] = 320,
        config: Optional[NeuroSightConfig] = None,
    ):
        """
        Initialize NeuroSight motion detector.
        
        Args:
            connectome_path: Optional path to a connectome weights JSON file.
                ``None`` (default) uses the weights bundled with the package.
            tile_size: Motion detection tile size (8 or 16)
            resize_width: Resize video to this width for efficiency (None = no resize)
            config: Optional NeuroSightConfig preset/configuration
        """
        self.detector = MotionDetector(
            connectome_weights_path=connectome_path,
            tile_size=tile_size,
            config=config,
        )
        self.video_processor = VideoProcessor(resize_width=resize_width)
        self.performance = PerformanceMonitor()
        
        logger.info("NeuroSightDetector initialized")

    def process_video(
        self,
        video_path: str,
        output_video: Optional[str] = None,
        overlay_vectors: bool = False,
        verbose: bool = True,
        return_vectors: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Process entire video, detect motion in all frames.
        
        Args:
            video_path: Path to input video
            output_video: If provided, write annotated video to this path
            overlay_vectors: Draw motion vectors on output frames
            verbose: Print progress
            return_vectors: Retain per-frame tile vectors in the returned dict.
                Off by default because retention grows linearly with video length.
        
        Returns:
            (aggregate_saliency_map, aggregate_motion_data)
            ``aggregate_saliency_map`` is the per-pixel maximum over all
            processed frames, not the last frame. ``aggregate_motion_data`` is
            empty unless ``return_vectors=True``.
        """
        output_writer = None
        all_motion_vectors: Dict[int, dict] = {}
        # A new video must not inherit temporal state from the previous one.
        self.detector.reset()
        self.performance = PerformanceMonitor()
        aggregate_saliency = None
        last_shape: Optional[Tuple[int, int]] = None
        output_fps = 30.0
        frame_dt = self.detector.config.default_dt
        
        try:
            frame_idx = 0
            
            for frame_idx, frame in self.video_processor.read_video(video_path):
                # Use the container's real frame rate when available so that
                # timestamps (and therefore the temporal low-pass) stay calibrated.
                props = self.video_processor.last_video_properties
                if props is not None and props.fps > 0:
                    output_fps = props.fps
                    frame_dt = props.frame_interval

                # Convert to grayscale for motion detection buffer (saves memory)
                gray_frame = self.video_processor.to_grayscale(frame)
                last_shape = gray_frame.shape[:2]
                
                # Stream frame into stateful detector
                start_time = self._get_time()
                # For video files we derive timestamps from the container FPS.
                # In real robotic use cases, timestamp would come from the camera
                # (or pass timestamp explicitly via MotionDetector.update).
                timestamp = frame_idx * frame_dt
                result = self.detector.update(gray_frame, timestamp=timestamp)
                latency = self._get_time() - start_time
                
                if result is None:
                    if verbose and frame_idx == 0:
                        logger.info("Warming up initial frame...")
                    continue
                    
                saliency = result.saliency_map
                # Building the per-tile dict is O(tiles) per frame; skip it unless used.
                need_vectors = return_vectors or overlay_vectors
                vectors = result.motion_vectors if need_vectors else {}
                
                self.performance.record_frame(latency)
                
                if aggregate_saliency is None:
                    aggregate_saliency = saliency.copy()
                else:
                    aggregate_saliency = np.maximum(aggregate_saliency, saliency)
                    
                if return_vectors:
                    all_motion_vectors[frame_idx] = vectors
                
                # Visualization (if requested)
                if output_video is not None:
                    if output_writer is None:
                        output_writer = OutputWriter(
                            output_video,
                            fps=output_fps,
                            frame_size=(frame.shape[1], frame.shape[0])
                        )
                    
                    # Create output frame (use original RGB frame, not the grayscale buffer frame)
                    output_frame = frame.copy()
                    
                    # Overlay saliency heatmap
                    saliency_colored = SaliencyMapVisualizer.apply_colormap(saliency)
                    # Blend saliency and original frame
                    output_frame = (0.5 * output_frame + 0.5 * saliency_colored).astype(np.uint8)
                    
                    # Overlay motion vectors if requested
                    if overlay_vectors:
                        output_frame = SaliencyMapVisualizer.overlay_vectors(
                            output_frame,
                            vectors,
                            tile_size=self.detector.tile_size,
                            arrow_scale=1.0
                        )
                    
                    output_writer.write_frame(output_frame)
                
                if verbose and (frame_idx + 1) % 30 == 0:
                    logger.info(f"Frame {frame_idx + 1}: {self.performance.report()}")
            
            if output_writer is not None:
                output_writer.release()
            
            if aggregate_saliency is None:
                if last_shape is None:
                    raise IOError(f"No frames could be read from video: {video_path}")
                logger.warning(
                    "Video produced no motion frames (fewer than 2 frames); "
                    "returning a zero saliency map."
                )
                aggregate_saliency = np.zeros(last_shape, dtype=np.float32)

            if verbose:
                logger.info(f"Video processing complete. {self.performance.report()}")
            
            return aggregate_saliency, all_motion_vectors
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        
        finally:
            if output_writer is not None:
                output_writer.release()

    def process_frame_pair(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame pair (for real-time streaming).
        
        Args:
            frame_prev: Previous frame (H, W) or (H, W, 3)
            frame_curr: Current frame (H, W) or (H, W, 3)
        
        Returns:
            (saliency_map, motion_vectors)
        """
        return self.detector.process_frame(frame_prev, frame_curr)

    def process_pair(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        timestamp: Optional[float] = None
    ) -> MotionResult:
        """
        Process a single frame pair and return a structured MotionResult.

        This is the forward-compatible API for benchmarks, robotics outputs, and
        richer diagnostics. `process_frame_pair()` remains available for legacy
        callers.
        """
        return self.detector.process_pair(frame_prev, frame_curr, timestamp=timestamp)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance metrics."""
        return {
            'fps': self.performance.get_fps(),
            'latency_ms': self.performance.get_latency_ms(),
            'total_frames': self.performance.frame_count
        }

    @staticmethod
    def _get_time() -> float:
        """Get current time in seconds."""
        import time
        return time.time()


# Example usage
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroSight Detector")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", help="Path to output video", default=None)
    parser.add_argument("--vectors", action="store_true", help="Overlay motion vectors on output video")
    
    args = parser.parse_args()
    
    # Initialize and process
    detector = NeuroSightDetector(
        connectome_path=None,
        tile_size=8,
        resize_width=320
    )
    
    saliency, vectors = detector.process_video(
        args.video_path,
        output_video=args.output,
        overlay_vectors=args.vectors,
        verbose=True
    )
    
    stats = detector.get_performance_stats()
    print(f"\nFinal Performance: {stats}")
