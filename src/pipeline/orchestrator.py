"""
SwimVision Pipeline Orchestrator
Coordinates all components for unified video processing pipeline.

This module provides the main pipeline that integrates:
- Pose estimation (RTMPose, MediaPipe, OpenPose, etc.)
- Multi-swimmer tracking (ByteTrack)
- Format conversions
- Multi-model fusion
- Results aggregation

Author: SwimVision Pro Team
Date: 2025-01-20
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Import pose estimators
from src.pose.model_registry import ModelRegistry
from src.pose.rtmpose_estimator import RTMPoseEstimator

# Import tracking
from src.tracking.bytetrack_tracker import ByteTrackTracker

# Import device utilities
from src.utils.device_utils import get_optimal_device

# Import format converters
from src.utils.format_converters import FormatConverter

# Import multi-model fusion (from existing codebase)
try:
    from src.pose.multi_model_fusion import MultiModelFusion
except ImportError:
    MultiModelFusion = None
    logging.warning("MultiModelFusion not available. Using single-model mode.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Pipeline processing modes."""

    REALTIME = "realtime"  # Optimize for speed
    BALANCED = "balanced"  # Balance speed and accuracy
    ACCURACY = "accuracy"  # Optimize for accuracy
    BATCH = "batch"  # Batch processing mode


@dataclass
class PipelineConfig:
    """Configuration for SwimVision pipeline."""

    # Pose estimation
    pose_models: list[str] = field(default_factory=lambda: ["rtmpose-m"])
    use_fusion: bool = False  # Use multi-model fusion

    # Tracking
    enable_tracking: bool = True
    track_thresh: float = 0.5  # High confidence threshold
    match_thresh: float = 0.7  # IoU matching threshold
    max_time_lost: int = 30  # Max frames before removing track

    # Processing mode
    mode: ProcessingMode = ProcessingMode.BALANCED

    # Format conversion
    output_formats: list[str] = field(default_factory=lambda: ["coco17"])  # coco17, smpl24, opensim

    # Visualization
    visualize: bool = True
    show_tracking_ids: bool = True
    show_keypoints: bool = True
    show_skeleton: bool = True

    # Performance
    device: str = "auto"  # cuda, mps, cpu, or auto for auto-detection
    batch_size: int = 1

    # Output
    save_results: bool = False
    output_dir: Path | None = None

    # Advanced features (placeholders for future phases)
    enable_underwater_preprocessing: bool = False
    enable_biomechanics: bool = False
    enable_3d_reconstruction: bool = False


@dataclass
class FrameResult:
    """Results from processing a single frame."""

    frame_id: int
    timestamp: float

    # Raw pose detections (before tracking)
    raw_poses: list[dict[str, Any]]  # Each dict has 'keypoints', 'bbox', 'score'

    # Tracked swimmers
    tracked_swimmers: list[dict[str, Any]]  # Each dict has 'track_id', 'keypoints', 'bbox', etc.

    # Converted formats
    converted_poses: dict[str, Any]  # e.g., {'smpl24': [...], 'opensim': [...]}

    # Performance metrics
    processing_time: float
    fps: float

    # Visualization
    visualized_frame: np.ndarray | None = None


class SwimVisionPipeline:
    """
    Main pipeline orchestrator for SwimVision Pro.

    Integrates pose estimation, tracking, and format conversion into
    a unified processing pipeline.

    Example:
        config = PipelineConfig(
            pose_models=["rtmpose-m"],
            enable_tracking=True,
            mode=ProcessingMode.REALTIME
        )
        pipeline = SwimVisionPipeline(config)

        # Process video
        for frame in video_frames:
            result = pipeline.process_frame(frame)
            cv2.imshow("SwimVision", result.visualized_frame)
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Resolve device (auto-detect if needed)
        if config.device == "auto":
            self.config.device = get_optimal_device()
        else:
            self.config.device = get_optimal_device(preferred=config.device)

        self.model_registry = ModelRegistry()

        # Initialize components
        self.pose_estimators = self._init_pose_estimators()
        self.tracker = self._init_tracker() if config.enable_tracking else None
        self.fusion = self._init_fusion() if config.use_fusion else None

        # Frame counter
        self.frame_count = 0
        self.start_time = time.time()

        # Results cache
        self.results_history: list[FrameResult] = []

        logger.info(
            f"SwimVision Pipeline initialized with {len(self.pose_estimators)} pose estimator(s)"
        )
        logger.info(f"Tracking: {'Enabled' if self.tracker else 'Disabled'}")
        logger.info(f"Mode: {config.mode.value}")

    def _init_pose_estimators(self) -> dict[str, Any]:
        """Initialize pose estimation models."""
        estimators = {}

        for model_name in self.config.pose_models:
            try:
                if model_name.startswith("rtmpose"):
                    # Use RTMPose
                    variant = model_name.split("-")[1]  # e.g., "m" from "rtmpose-m"
                    estimators[model_name] = RTMPoseEstimator(
                        variant=variant, device=self.config.device
                    )
                    logger.info(f"Loaded {model_name}")

                else:
                    # Use model registry for other models
                    model = self.model_registry.load_model(model_name, device=self.config.device)
                    estimators[model_name] = model
                    logger.info(f"Loaded {model_name}")

            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")

        if not estimators:
            raise RuntimeError("No pose estimators could be loaded!")

        return estimators

    def _init_tracker(self) -> ByteTrackTracker | None:
        """Initialize ByteTrack tracker."""
        if not self.config.enable_tracking:
            return None

        try:
            tracker = ByteTrackTracker(
                track_thresh=self.config.track_thresh,
                match_thresh=self.config.match_thresh,
                max_time_lost=self.config.max_time_lost,
            )
            logger.info("ByteTrack tracker initialized")
            return tracker
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            return None

    def _init_fusion(self) -> Any | None:
        """Initialize multi-model fusion."""
        if not self.config.use_fusion or MultiModelFusion is None:
            return None

        try:
            # Use existing fusion from previous implementation
            fusion = MultiModelFusion()
            logger.info("Multi-model fusion enabled")
            return fusion
        except Exception as e:
            logger.error(f"Failed to initialize fusion: {e}")
            return None

    def process_frame(self, frame: np.ndarray, frame_id: int | None = None) -> FrameResult:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input frame (BGR format)
            frame_id: Optional frame ID (auto-incremented if not provided)

        Returns:
            FrameResult containing all processing results
        """
        start_time = time.time()

        if frame_id is None:
            frame_id = self.frame_count
        self.frame_count += 1

        # Step 1: Pose estimation
        raw_poses = self._estimate_poses(frame)

        # Step 2: Multi-model fusion (if enabled)
        if self.fusion and len(self.pose_estimators) > 1:
            raw_poses = self._fuse_poses(raw_poses)

        # Step 3: Tracking (if enabled)
        tracked_swimmers = self._track_swimmers(raw_poses, frame_id)

        # Step 4: Format conversion
        converted_poses = self._convert_formats(tracked_swimmers or raw_poses)

        # Step 5: Visualization (if enabled)
        visualized_frame = None
        if self.config.visualize:
            visualized_frame = self._visualize_results(frame.copy(), tracked_swimmers or raw_poses)

        # Calculate performance metrics
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        # Create result
        result = FrameResult(
            frame_id=frame_id,
            timestamp=time.time() - self.start_time,
            raw_poses=raw_poses,
            tracked_swimmers=tracked_swimmers if tracked_swimmers else raw_poses,
            converted_poses=converted_poses,
            processing_time=processing_time,
            fps=fps,
            visualized_frame=visualized_frame,
        )

        # Cache result
        self.results_history.append(result)

        return result

    def _estimate_poses(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Run pose estimation on frame.

        Returns:
            List of pose detections, each with 'keypoints', 'bbox', 'score'
        """
        all_poses = []

        for model_name, estimator in self.pose_estimators.items():
            try:
                # Run pose estimation
                result = estimator.estimate_pose(frame, return_image=False)

                if result is not None:
                    keypoints = result.get("keypoints", [])
                    scores = result.get("scores", [])
                    bboxes = result.get("bboxes", [])

                    # Convert to standard format
                    if len(keypoints) > 0:
                        for i, kpts in enumerate(keypoints):
                            pose = {
                                "keypoints": kpts,
                                "score": scores[i] if i < len(scores) else 0.0,
                                "bbox": bboxes[i] if i < len(bboxes) else self._estimate_bbox(kpts),
                                "model": model_name,
                                "format": "coco17",  # RTMPose uses COCO format
                            }
                            all_poses.append(pose)

            except Exception as e:
                logger.error(f"Pose estimation failed for {model_name}: {e}")

        return all_poses

    def _estimate_bbox(self, keypoints: np.ndarray) -> np.ndarray:
        """Estimate bounding box from keypoints."""
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]  # Filter by confidence
        if len(valid_kpts) == 0:
            return np.array([0, 0, 100, 100, 0.0])

        x_min = valid_kpts[:, 0].min()
        y_min = valid_kpts[:, 1].min()
        x_max = valid_kpts[:, 0].max()
        y_max = valid_kpts[:, 1].max()

        # Add padding
        w = x_max - x_min
        h = y_max - y_min
        x_min = max(0, x_min - 0.1 * w)
        y_min = max(0, y_min - 0.1 * h)
        x_max = x_max + 0.1 * w
        y_max = y_max + 0.1 * h

        score = valid_kpts[:, 2].mean()

        return np.array([x_min, y_min, x_max, y_max, score])

    def _fuse_poses(self, poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fuse poses from multiple models."""
        if not self.fusion or len(poses) <= 1:
            return poses

        try:
            # Group poses by person (using bbox IoU)
            # Then fuse keypoints from different models
            # This is a placeholder - full implementation in multi_model_fusion.py
            return poses
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            return poses

    def _track_swimmers(
        self, poses: list[dict[str, Any]], frame_id: int
    ) -> list[dict[str, Any]] | None:
        """
        Track swimmers across frames.

        Returns:
            List of tracked swimmers with track_id added
        """
        if not self.tracker:
            return None

        try:
            # Convert poses to dict format for ByteTrack
            detections = []
            for pose in poses:
                bbox = pose["bbox"]
                score = pose.get("score", pose.get("confidence", 0.0))
                keypoints = pose.get("keypoints")

                # ByteTrackTracker.update() expects list[dict]
                det = {
                    "bbox": bbox[:4] if len(bbox) >= 4 else bbox,
                    "confidence": float(score),
                    "keypoints": keypoints,
                }
                detections.append(det)

            # Update tracker
            tracks = self.tracker.update(detections, frame_id=frame_id)

            # Convert tracks back to pose format with track_id
            tracked_poses = []
            for track in tracks:
                tracked_pose = {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "keypoints": track.keypoints,
                    "confidence": track.confidence,
                    "state": track.state.value,
                    "trajectory": list(track.bbox_history)[-10:]
                    if hasattr(track, "bbox_history")
                    else [],
                    "velocity": track.get_velocity(),
                    "format": "coco17",
                }
                tracked_poses.append(tracked_pose)

            return tracked_poses

        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return None

    def _convert_formats(self, poses: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """
        Convert poses to requested output formats.

        Returns:
            Dict mapping format name to converted poses
        """
        converted = {}

        for output_format in self.config.output_formats:
            if output_format == "coco17":
                # Already in COCO format
                converted[output_format] = [p["keypoints"] for p in poses]

            elif output_format == "smpl24":
                # Convert COCO-17 to SMPL-24
                smpl_poses = []
                for pose in poses:
                    kpts_coco = pose["keypoints"]
                    kpts_smpl = FormatConverter.coco17_to_smpl24(kpts_coco)
                    smpl_poses.append(kpts_smpl)
                converted[output_format] = smpl_poses

            elif output_format == "opensim":
                # Convert COCO-17 -> SMPL-24 -> OpenSim
                opensim_poses = []
                for pose in poses:
                    kpts_coco = pose["keypoints"]
                    kpts_smpl = FormatConverter.coco17_to_smpl24(kpts_coco)
                    markers = FormatConverter.smpl24_to_opensim_markers(kpts_smpl)
                    opensim_poses.append(markers)
                converted[output_format] = opensim_poses

        return converted

    def _visualize_results(self, frame: np.ndarray, poses: list[dict[str, Any]]) -> np.ndarray:
        """
        Visualize poses and tracking on frame.

        Args:
            frame: Frame to draw on
            poses: List of poses (with optional track_id)

        Returns:
            Annotated frame
        """
        # COCO-17 skeleton connections
        skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),  # Arms
            (5, 11),
            (6, 12),
            (11, 12),  # Torso
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Legs
        ]

        # Color palette for tracks
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 128),
            (255, 128, 0),
            (0, 128, 255),
        ]

        for pose in poses:
            keypoints = pose.get("keypoints")
            track_id = pose.get("track_id")
            bbox = pose.get("bbox")

            if keypoints is None:
                continue

            # Choose color based on track_id
            if track_id is not None:
                color = colors[track_id % len(colors)]
            else:
                color = (0, 255, 0)

            # Draw bounding box
            if bbox is not None and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw track ID
                if track_id is not None and self.config.show_tracking_ids:
                    label = f"ID: {track_id}"
                    cv2.putText(
                        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

            # Draw skeleton
            if self.config.show_skeleton:
                for i, j in skeleton:
                    if i < len(keypoints) and j < len(keypoints):
                        pt1 = keypoints[i]
                        pt2 = keypoints[j]

                        # Check confidence
                        if pt1[2] > 0.3 and pt2[2] > 0.3:
                            x1, y1 = int(pt1[0]), int(pt1[1])
                            x2, y2 = int(pt2[0]), int(pt2[1])
                            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            # Draw keypoints
            if self.config.show_keypoints:
                for i, kpt in enumerate(keypoints):
                    if kpt[2] > 0.3:  # Confidence threshold
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(frame, (x, y), 3, color, -1)

        # Draw performance info
        if len(self.results_history) > 0:
            last_result = self.results_history[-1]
            fps_text = f"FPS: {last_result.fps:.1f}"
            swimmers_text = f"Swimmers: {len(poses)}"

            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                frame, swimmers_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        return frame

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        show_preview: bool = True,
    ) -> list[FrameResult]:
        """
        Process entire video file.

        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            show_preview: Show live preview window

        Returns:
            List of FrameResult for all frames
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

        # Setup video writer if output requested
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        results = []
        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                result = self.process_frame(frame, frame_id=frame_id)
                results.append(result)

                # Write output
                if writer and result.visualized_frame is not None:
                    writer.write(result.visualized_frame)

                # Show preview
                if show_preview and result.visualized_frame is not None:
                    cv2.imshow("SwimVision Pipeline", result.visualized_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Preview stopped by user")
                        break

                # Progress update
                if frame_id % 30 == 0:
                    progress = (frame_id / total_frames) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% "
                        f"({frame_id}/{total_frames}), "
                        f"FPS: {result.fps:.1f}"
                    )

                frame_id += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        logger.info(f"Processed {len(results)} frames")
        if output_path:
            logger.info(f"Output saved to: {output_path}")

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        if not self.results_history:
            return {}

        processing_times = [r.processing_time for r in self.results_history]
        fps_values = [r.fps for r in self.results_history]
        swimmer_counts = [len(r.tracked_swimmers) for r in self.results_history]

        return {
            "total_frames": len(self.results_history),
            "avg_processing_time": np.mean(processing_times),
            "avg_fps": np.mean(fps_values),
            "max_fps": np.max(fps_values),
            "min_fps": np.min(fps_values),
            "avg_swimmers_per_frame": np.mean(swimmer_counts),
            "max_swimmers": np.max(swimmer_counts) if swimmer_counts else 0,
            "total_runtime": self.results_history[-1].timestamp if self.results_history else 0,
        }

    def reset(self):
        """Reset pipeline state."""
        self.frame_count = 0
        self.start_time = time.time()
        self.results_history.clear()

        if self.tracker:
            self.tracker.reset()

        logger.info("Pipeline reset")


def demo_pipeline():
    """Demo the pipeline on a test video."""
    # Configuration
    config = PipelineConfig(
        pose_models=["rtmpose-m"],
        enable_tracking=True,
        mode=ProcessingMode.REALTIME,
        output_formats=["coco17", "smpl24"],
        visualize=True,
        show_tracking_ids=True,
        device="cuda",
    )

    # Initialize pipeline
    pipeline = SwimVisionPipeline(config)

    # Test video path (update this to your test video)
    test_video = "data/videos/swimming_test.mp4"

    if Path(test_video).exists():
        # Process video
        results = pipeline.process_video(
            test_video, output_path="results/pipeline_output.mp4", show_preview=True
        )

        # Print statistics
        stats = pipeline.get_statistics()
        print("\n" + "=" * 50)
        print("PIPELINE STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print(f"Test video not found: {test_video}")
        print("Please provide a swimming video at data/videos/swimming_test.mp4")


if __name__ == "__main__":
    demo_pipeline()
