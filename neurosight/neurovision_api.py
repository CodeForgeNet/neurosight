"""
Phase 4: NeuroSight Detector API
High-level interface: user processes video in 3 lines of code.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import logging

from .detector import MotionDetector
from .utils import FrameBuffer, VideoProcessor, PerformanceMonitor, SaliencyMapVisualizer, OutputWriter

logger = logging.getLogger(__name__)


class NeuroSightDetector:
    """
    User-facing API for NeuroSight motion detection.
    
    Example:
        detector = NeuroSightDetector(connectome_path="connectome_weights.json")
        saliency, vectors = detector.process_video("input.mp4", output_video="output.mp4")
    """

    def __init__(
        self,
        connectome_path: str = "connectome_weights.json",
        tile_size: int = 8,
        resize_width: Optional[int] = 320
    ):
        """
        Initialize NeuroSight motion detector.
        
        Args:
            connectome_path: Path to connectome_weights.json (from Phase 1)
            tile_size: Motion detection tile size (8 or 16)
            resize_width: Resize video to this width for efficiency (None = no resize)
        """
        self.detector = MotionDetector(
            connectome_weights_path=connectome_path,
            tile_size=tile_size
        )
        self.video_processor = VideoProcessor(resize_width=resize_width)
        self.frame_buffer = FrameBuffer(buffer_size=2)
        self.performance = PerformanceMonitor()
        
        logger.info("NeuroSightDetector initialized")

    def process_video(
        self,
        video_path: str,
        output_video: Optional[str] = None,
        overlay_vectors: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Process entire video, detect motion in all frames.
        
        Args:
            video_path: Path to input video
            output_video: If provided, write annotated video to this path
            overlay_vectors: Draw motion vectors on output frames
            verbose: Print progress
        
        Returns:
            (final_saliency_map, aggregate_motion_data)
        """
        output_writer = None
        all_motion_vectors = {}
        final_saliency = None
        
        try:
            frame_idx = 0
            
            for frame_idx, frame in self.video_processor.read_video(video_path):
                # Convert to grayscale for motion detection buffer (saves memory)
                gray_frame = self.video_processor.to_grayscale(frame)
                
                # Push frame to buffer
                is_ready = self.frame_buffer.push(gray_frame)
                
                if not is_ready:
                    if verbose and frame_idx == 0:
                        logger.info("Buffering initial frame...")
                    continue
                
                # Process frame pair
                frame_prev, frame_curr = self.frame_buffer.get_pair()
                
                start_time = self._get_time()
                saliency, vectors = self.detector.process_frame(frame_prev, frame_curr)
                latency = self._get_time() - start_time
                
                self.performance.record_frame(latency)
                
                if final_saliency is None:
                    final_saliency = saliency.copy()
                else:
                    final_saliency = np.maximum(final_saliency, saliency)
                    
                all_motion_vectors[frame_idx] = vectors
                
                # Visualization (if requested)
                if output_video is not None:
                    if output_writer is None:
                        output_writer = OutputWriter(
                            output_video,
                            fps=30.0,
                            frame_size=(frame_curr.shape[1], frame_curr.shape[0])
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
            
            if verbose:
                logger.info(f"Video processing complete. {self.performance.report()}")
            
            return final_saliency, all_motion_vectors
        
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
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroSight Detector")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", help="Path to output video", default=None)
    parser.add_argument("--vectors", action="store_true", help="Overlay motion vectors on output video")
    
    args = parser.parse_args()
    
    # Initialize and process
    detector = NeuroSightDetector(
        connectome_path="connectome_weights.json",
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
