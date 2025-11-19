"""Model comparison and benchmarking tools.

Compare performance of different pose estimation models on:
- Accuracy metrics
- Speed/FPS
- Keypoint detection rate
- Confidence scores
- Robustness to occlusions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np
import cv2
import json

from src.pose.base_estimator import BasePoseEstimator, KeypointFormat


@dataclass
class ModelMetrics:
    """Performance metrics for a single model."""
    model_name: str
    avg_inference_time: float = 0.0        # Average time per frame (seconds)
    fps: float = 0.0                       # Frames per second
    avg_confidence: float = 0.0            # Average keypoint confidence
    detection_rate: float = 0.0            # % of frames with detections
    avg_keypoints_detected: float = 0.0    # Average number of keypoints per frame
    keypoint_visibility: Dict[str, float] = field(default_factory=dict)  # Per-keypoint visibility rate
    memory_usage_mb: float = 0.0           # Approximate memory usage
    total_frames_processed: int = 0
    failed_frames: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results of model comparison."""
    models_compared: List[str]
    metrics: Dict[str, ModelMetrics]       # Model name -> metrics
    winner_by_metric: Dict[str, str]       # Metric name -> best model
    frame_by_frame_comparison: List[Dict]  # Per-frame comparison data
    summary: str                            # Summary text
    timestamp: str                          # When comparison was run


class ModelComparison:
    """Compare multiple pose estimation models."""

    def __init__(
        self,
        models: Dict[str, BasePoseEstimator],
        output_dir: str = "results/model_comparison",
    ):
        """Initialize model comparison.

        Args:
            models: Dictionary of {model_name: estimator}.
            output_dir: Directory for output results.
        """
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics = {name: ModelMetrics(model_name=name) for name in models.keys()}

        # Frame-by-frame comparison data
        self.frame_comparisons = []

    def benchmark_on_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        visualize: bool = False,
    ) -> ComparisonResult:
        """Benchmark all models on a video.

        Args:
            video_path: Path to test video.
            max_frames: Maximum frames to process (None for all).
            visualize: Whether to save visualization frames.

        Returns:
            Comparison results.
        """
        print(f"Benchmarking models on {video_path}...")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        frame_idx = 0

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process with all models
            frame_results = self._process_frame_all_models(frame, frame_idx)
            self.frame_comparisons.append(frame_results)

            if visualize and frame_idx % 30 == 0:  # Save every 30th frame
                self._visualize_comparison(frame, frame_results, frame_idx)

            frame_idx += 1

            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")

        cap.release()

        # Calculate final metrics
        self._calculate_final_metrics()

        # Determine winners
        winners = self._determine_winners()

        # Generate summary
        summary = self._generate_summary()

        # Create result
        from datetime import datetime
        result = ComparisonResult(
            models_compared=list(self.models.keys()),
            metrics=self.metrics,
            winner_by_metric=winners,
            frame_by_frame_comparison=self.frame_comparisons,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )

        # Save results
        self._save_results(result)

        return result

    def _process_frame_all_models(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> Dict:
        """Process single frame with all models.

        Args:
            frame: Input frame.
            frame_idx: Frame index.

        Returns:
            Dictionary with results from all models.
        """
        results = {
            'frame_idx': frame_idx,
            'models': {},
        }

        for model_name, model in self.models.items():
            try:
                # Time inference
                start_time = time.time()
                pose_data, _ = model.estimate_pose(frame, return_image=False)
                inference_time = time.time() - start_time

                # Extract metrics
                if pose_data is not None:
                    # Handle multi-person detection (take first person)
                    if isinstance(pose_data, list):
                        pose_data = pose_data[0] if len(pose_data) > 0 else None

                    if pose_data is not None:
                        keypoints = pose_data.get('keypoints', np.array([]))
                        confidences = keypoints[:, 2] if len(keypoints) > 0 else np.array([])

                        results['models'][model_name] = {
                            'detected': True,
                            'inference_time': inference_time,
                            'num_keypoints': len(keypoints),
                            'avg_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
                            'keypoints': keypoints.tolist(),
                            'keypoint_names': pose_data.get('keypoint_names', []),
                        }

                        # Update running metrics
                        metrics = self.metrics[model_name]
                        metrics.total_frames_processed += 1

                        # Update averages (running average)
                        n = metrics.total_frames_processed
                        metrics.avg_inference_time = (metrics.avg_inference_time * (n-1) + inference_time) / n
                        metrics.avg_confidence = (metrics.avg_confidence * (n-1) + results['models'][model_name]['avg_confidence']) / n
                        metrics.avg_keypoints_detected = (metrics.avg_keypoints_detected * (n-1) + len(keypoints)) / n

                        # Update per-keypoint visibility
                        for kp_name, kp in zip(pose_data.get('keypoint_names', []), keypoints):
                            if kp_name not in metrics.keypoint_visibility:
                                metrics.keypoint_visibility[kp_name] = 0.0

                            # Running average of visibility (conf > 0.3)
                            visible = 1.0 if kp[2] > 0.3 else 0.0
                            metrics.keypoint_visibility[kp_name] = (
                                metrics.keypoint_visibility[kp_name] * (n-1) + visible
                            ) / n

                        continue

                # No detection
                results['models'][model_name] = {
                    'detected': False,
                    'inference_time': inference_time,
                }

                metrics = self.metrics[model_name]
                metrics.failed_frames += 1

            except Exception as e:
                print(f"Error processing frame {frame_idx} with {model_name}: {e}")
                results['models'][model_name] = {
                    'detected': False,
                    'error': str(e),
                }
                self.metrics[model_name].failed_frames += 1

        return results

    def _calculate_final_metrics(self):
        """Calculate final aggregate metrics."""
        for model_name, metrics in self.metrics.items():
            # Calculate FPS
            if metrics.avg_inference_time > 0:
                metrics.fps = 1.0 / metrics.avg_inference_time

            # Calculate detection rate
            total_attempts = metrics.total_frames_processed + metrics.failed_frames
            if total_attempts > 0:
                metrics.detection_rate = metrics.total_frames_processed / total_attempts

    def _determine_winners(self) -> Dict[str, str]:
        """Determine best model for each metric.

        Returns:
            Dictionary of {metric_name: best_model_name}.
        """
        winners = {}

        # Best FPS (higher is better)
        best_fps_model = max(self.metrics.items(), key=lambda x: x[1].fps)
        winners['fps'] = best_fps_model[0]

        # Best average confidence (higher is better)
        best_conf_model = max(self.metrics.items(), key=lambda x: x[1].avg_confidence)
        winners['confidence'] = best_conf_model[0]

        # Best detection rate (higher is better)
        best_det_model = max(self.metrics.items(), key=lambda x: x[1].detection_rate)
        winners['detection_rate'] = best_det_model[0]

        # Best average keypoints detected (higher is better)
        best_kp_model = max(self.metrics.items(), key=lambda x: x[1].avg_keypoints_detected)
        winners['keypoints_detected'] = best_kp_model[0]

        return winners

    def _generate_summary(self) -> str:
        """Generate text summary of comparison.

        Returns:
            Summary string.
        """
        lines = ["Model Comparison Summary", "=" * 50, ""]

        for model_name, metrics in self.metrics.items():
            lines.append(f"{model_name}:")
            lines.append(f"  FPS: {metrics.fps:.2f}")
            lines.append(f"  Avg Inference Time: {metrics.avg_inference_time*1000:.2f}ms")
            lines.append(f"  Detection Rate: {metrics.detection_rate*100:.1f}%")
            lines.append(f"  Avg Confidence: {metrics.avg_confidence:.3f}")
            lines.append(f"  Avg Keypoints: {metrics.avg_keypoints_detected:.1f}")
            lines.append(f"  Frames Processed: {metrics.total_frames_processed}")
            lines.append(f"  Failed Frames: {metrics.failed_frames}")
            lines.append("")

        # Winners
        winners = self._determine_winners()
        lines.append("Best Models by Metric:")
        lines.append("-" * 50)
        for metric, model in winners.items():
            lines.append(f"  {metric}: {model}")

        return "\n".join(lines)

    def _visualize_comparison(
        self,
        frame: np.ndarray,
        frame_results: Dict,
        frame_idx: int,
    ):
        """Create visualization comparing all models on this frame.

        Args:
            frame: Input frame.
            frame_results: Results from all models.
            frame_idx: Frame index.
        """
        # Create grid of model outputs
        num_models = len(self.models)
        grid_size = int(np.ceil(np.sqrt(num_models)))

        h, w = frame.shape[:2]
        grid = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.uint8)

        for idx, (model_name, result) in enumerate(frame_results['models'].items()):
            row = idx // grid_size
            col = idx % grid_size

            model_frame = frame.copy()

            # Draw keypoints if detected
            if result.get('detected', False):
                keypoints = np.array(result['keypoints'])
                for kp in keypoints:
                    if kp[2] > 0.3:  # Confidence threshold
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(model_frame, (x, y), 5, (0, 255, 0), -1)

            # Add model name and metrics
            cv2.putText(model_frame, model_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if 'inference_time' in result:
                fps_text = f"FPS: {1.0/result['inference_time']:.1f}"
                cv2.putText(model_frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if 'avg_confidence' in result:
                conf_text = f"Conf: {result['avg_confidence']:.2f}"
                cv2.putText(model_frame, conf_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Place in grid
            y1 = row * h
            y2 = (row + 1) * h
            x1 = col * w
            x2 = (col + 1) * w

            grid[y1:y2, x1:x2] = model_frame

        # Save visualization
        vis_path = self.output_dir / f"comparison_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(vis_path), grid)

    def _save_results(self, result: ComparisonResult):
        """Save comparison results to files.

        Args:
            result: Comparison results.
        """
        # Save summary text
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(result.summary)

        # Save detailed metrics as JSON
        metrics_path = self.output_dir / "metrics.json"
        metrics_dict = {}
        for model_name, metrics in result.metrics.items():
            metrics_dict[model_name] = {
                'avg_inference_time': metrics.avg_inference_time,
                'fps': metrics.fps,
                'avg_confidence': metrics.avg_confidence,
                'detection_rate': metrics.detection_rate,
                'avg_keypoints_detected': metrics.avg_keypoints_detected,
                'keypoint_visibility': metrics.keypoint_visibility,
                'total_frames_processed': metrics.total_frames_processed,
                'failed_frames': metrics.failed_frames,
            }

        with open(metrics_path, 'w') as f:
            json.dump({
                'models_compared': result.models_compared,
                'metrics': metrics_dict,
                'winner_by_metric': result.winner_by_metric,
                'timestamp': result.timestamp,
            }, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")
        print(f"Summary: {summary_path}")
        print(f"Metrics: {metrics_path}")

    def generate_comparison_report(self) -> str:
        """Generate detailed comparison report.

        Returns:
            Markdown report.
        """
        lines = ["# Pose Estimation Model Comparison Report", ""]

        # Overview
        lines.append("## Overview")
        lines.append(f"Models Compared: {', '.join(self.models.keys())}")
        lines.append("")

        # Performance Table
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("| Model | FPS | Avg Time (ms) | Detection Rate | Avg Confidence | Avg Keypoints |")
        lines.append("|-------|-----|---------------|----------------|----------------|---------------|")

        for model_name, metrics in self.metrics.items():
            lines.append(
                f"| {model_name} | {metrics.fps:.2f} | {metrics.avg_inference_time*1000:.2f} | "
                f"{metrics.detection_rate*100:.1f}% | {metrics.avg_confidence:.3f} | "
                f"{metrics.avg_keypoints_detected:.1f} |"
            )

        lines.append("")

        # Winners
        winners = self._determine_winners()
        lines.append("## Best Models")
        lines.append("")
        for metric, model in winners.items():
            lines.append(f"- **{metric}**: {model}")

        lines.append("")

        # Per-Keypoint Visibility
        lines.append("## Keypoint Visibility by Model")
        lines.append("")

        for model_name, metrics in self.metrics.items():
            if metrics.keypoint_visibility:
                lines.append(f"### {model_name}")
                lines.append("")

                sorted_kps = sorted(
                    metrics.keypoint_visibility.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for kp_name, visibility in sorted_kps[:10]:  # Top 10
                    lines.append(f"- {kp_name}: {visibility*100:.1f}%")

                lines.append("")

        return "\n".join(lines)


def create_comparison_from_config(config_path: str) -> ModelComparison:
    """Create model comparison from configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        ModelComparison instance.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = {}

    for model_config in config.get('models', []):
        model_type = model_config['type']
        model_name = model_config['name']

        if model_type == 'yolo':
            from src.pose.yolo_estimator import YOLOPoseEstimator
            models[model_name] = YOLOPoseEstimator(
                model_config.get('variant', 'yolo11n-pose.pt'),
                model_config.get('device', 'cpu'),
                model_config.get('confidence', 0.5),
            )

        elif model_type == 'mediapipe':
            from src.pose.mediapipe_estimator import MediaPipeEstimator
            models[model_name] = MediaPipeEstimator(
                model_config.get('complexity', 1),
                model_config.get('confidence', 0.5),
                model_config.get('device', 'cpu'),
            )

        # Add more model types as needed

    return ModelComparison(models, config.get('output_dir', 'results/comparison'))
