"""Demonstration of advanced pose estimation models.

Shows how to use:
- MediaPipe pose estimator
- Multi-model fusion
- Water surface detection
- Adaptive threshold tuning
- Model comparison
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def demo_mediapipe():
    """Demonstrate MediaPipe pose estimation."""
    print("\n" + "="*60)
    print("MediaPipe Pose Estimation Demo")
    print("="*60)

    try:
        from src.pose.mediapipe_estimator import MediaPipeEstimator

        # Initialize estimator
        print("\nInitializing MediaPipe (complexity=1)...")
        estimator = MediaPipeEstimator(
            model_complexity=1,
            min_detection_confidence=0.5,
        )

        print(f"Model loaded successfully!")
        print(f"  - Supports 3D: {estimator.supports_3d()}")
        print(f"  - Supports multi-person: {estimator.supports_multi_person()}")
        print(f"  - Keypoint format: {estimator.get_keypoint_format().value}")

        # Create test image (or load from file)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Estimate pose
        print("\nEstimating pose...")
        pose_data, annotated_image = estimator.estimate_pose(test_image)

        if pose_data:
            print(f"  - Detected {len(pose_data['keypoints'])} keypoints")
            print(f"  - Average confidence: {np.mean(pose_data['keypoints'][:, 2]):.3f}")

            # Get 3D landmarks
            landmarks_3d = estimator.get_3d_landmarks(pose_data)
            if landmarks_3d is not None:
                print(f"  - 3D landmarks available: {len(landmarks_3d)} points")

            # Convert to COCO-17
            coco17_data = estimator.convert_to_coco17(pose_data)
            print(f"  - Converted to COCO-17: {len(coco17_data['keypoints'])} keypoints")
        else:
            print("  - No pose detected (expected on blank image)")

        print("\nMediaPipe demo complete!")

    except ImportError:
        print("ERROR: MediaPipe not installed. Install with: pip install mediapipe")


def demo_model_fusion():
    """Demonstrate multi-model fusion."""
    print("\n" + "="*60)
    print("Multi-Model Fusion Demo")
    print("="*60)

    try:
        from src.pose.model_fusion import MultiModelFusion, FusionMethod
        from src.pose.yolo_estimator import YOLOPoseEstimator

        print("\nInitializing models for fusion...")

        # Create models
        models = []

        # YOLO model
        try:
            yolo_model = YOLOPoseEstimator("yolo11n-pose.pt", device="cpu", confidence=0.5)
            models.append(yolo_model)
            print("  - Added YOLO11")
        except Exception as e:
            print(f"  - YOLO11 not available: {e}")

        # MediaPipe model
        try:
            from src.pose.mediapipe_estimator import MediaPipeEstimator
            mp_model = MediaPipeEstimator(model_complexity=1, min_detection_confidence=0.5)
            models.append(mp_model)
            print("  - Added MediaPipe")
        except ImportError:
            print("  - MediaPipe not available")

        if len(models) < 2:
            print("\nNeed at least 2 models for fusion. Skipping demo.")
            return

        # Create fusion
        print(f"\nCreating fusion system with {len(models)} models...")
        fusion = MultiModelFusion(
            models=models,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            confidence_threshold=0.3,
        )

        # Test fusion
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        print("\nRunning fusion...")
        fused_result, annotated = fusion.estimate_pose(test_image, return_image=False)

        if fused_result:
            print(f"  - Fused {len(fused_result.keypoints)} keypoints")
            print(f"  - Contributing models: {fused_result.contributing_models}")
            print(f"  - Fusion method: {fused_result.fusion_method.value}")
            print("\n  Confidence scores by model:")
            for model, conf in fused_result.confidence_scores.items():
                print(f"    {model}: {conf:.3f}")
        else:
            print("  - No detection (expected on blank image)")

        print("\nModel fusion demo complete!")

    except ImportError as e:
        print(f"ERROR: Required modules not available: {e}")


def demo_water_surface_detection():
    """Demonstrate water surface detection."""
    print("\n" + "="*60)
    print("Water Surface Detection Demo")
    print("="*60)

    from src.analysis.water_surface_detector import WaterSurfaceDetector, WaterState

    # Create simulated pool image
    print("\nCreating simulated pool image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add blue water in bottom half
    img[240:, :] = [200, 100, 50]  # Blue-ish color (BGR)

    # Add white edge at water surface
    img[235:245, :] = [255, 255, 255]

    # Initialize detector
    print("\nInitializing water surface detector...")
    detector = WaterSurfaceDetector(
        pool_type="indoor",
        detection_method="hybrid",
    )

    # Detect surface
    print("\nDetecting water surface...")
    surface_info = detector.detect_surface(img)

    if surface_info:
        print(f"  - Water level: {surface_info.water_level:.1f} pixels")
        print(f"  - Confidence: {surface_info.confidence:.3f}")
        print(f"  - Detection method: {surface_info.metadata.get('method')}")
        print(f"  - Surface points: {len(surface_info.surface_points)}")

        # Test water state detection
        print("\nTesting water state detection:")
        test_points = [
            ((320, 100), "Above water"),
            ((320, 240), "At surface"),
            ((320, 400), "Below water"),
        ]

        for point, description in test_points:
            state = detector.get_water_state(point, surface_info)
            print(f"  - Point {point}: {description} -> {state.value}")

        # Visualize
        print("\nCreating visualization...")
        visualized = detector.visualize_surface(img, surface_info)

        output_path = Path("results/water_surface_demo.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), visualized)
        print(f"  - Saved to: {output_path}")

    else:
        print("  - No surface detected")

    print("\nWater surface detection demo complete!")


def demo_adaptive_tuning():
    """Demonstrate adaptive threshold tuning."""
    print("\n" + "="*60)
    print("Adaptive Threshold Tuning Demo")
    print("="*60)

    from src.utils.adaptive_tuning import AdaptiveThresholdTuner, PoolCondition

    print("\nInitializing adaptive tuner...")
    tuner = AdaptiveThresholdTuner(
        auto_tune=True,
        adaptation_rate=0.1,
    )

    print(f"Initial confidence threshold: {tuner.params.confidence_threshold:.3f}")

    # Simulate detection sequence with varying conditions
    print("\nSimulating 50 frames with varying conditions...")

    for i in range(50):
        # Create mock detection with varying confidence
        if i < 20:
            # Poor conditions (low confidence)
            conf_range = (0.2, 0.4)
            description = "Poor"
        elif i < 40:
            # Good conditions
            conf_range = (0.6, 0.8)
            description = "Good"
        else:
            # Excellent conditions
            conf_range = (0.8, 0.95)
            description = "Excellent"

        conf = np.random.uniform(*conf_range)
        detection = {
            'keypoints': np.random.rand(17, 3) * conf,
        }

        params = tuner.update(detection, frame_stats={'brightness': 120})

        # Print every 10 frames
        if (i + 1) % 10 == 0:
            metrics = tuner.get_current_metrics()
            print(f"\nFrame {i+1} ({description} conditions):")
            print(f"  - Confidence threshold: {params.confidence_threshold:.3f}")
            print(f"  - Occlusion threshold: {params.occlusion_threshold:.3f}")
            print(f"  - Current condition: {metrics['current_condition']}")
            print(f"  - Avg confidence: {metrics['avg_confidence']:.3f}")
            print(f"  - Detection rate: {metrics['detection_rate']*100:.1f}%")

    print("\nAdaptive tuning demo complete!")


def demo_model_comparison():
    """Demonstrate model comparison tools."""
    print("\n" + "="*60)
    print("Model Comparison Demo")
    print("="*60)

    try:
        from src.utils.model_comparison import ModelComparison
        from src.pose.yolo_estimator import YOLOPoseEstimator

        print("\nInitializing models for comparison...")

        models = {}

        # Add YOLO variants
        for variant in ["yolo11n-pose.pt", "yolo11s-pose.pt"]:
            try:
                models[variant] = YOLOPoseEstimator(variant, device="cpu", confidence=0.5)
                print(f"  - Added {variant}")
            except Exception as e:
                print(f"  - {variant} not available: {e}")

        # Add MediaPipe
        try:
            from src.pose.mediapipe_estimator import MediaPipeEstimator
            models["MediaPipe"] = MediaPipeEstimator()
            print("  - Added MediaPipe")
        except ImportError:
            pass

        if len(models) < 2:
            print("\nNeed at least 2 models for comparison. Skipping demo.")
            return

        # Create comparison
        print(f"\nCreating comparison with {len(models)} models...")
        comparison = ModelComparison(models, output_dir="results/comparison_demo")

        # Generate sample video
        print("\nGenerating test video...")
        test_video_path = Path("data/videos/test_comparison.mp4")
        test_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Create simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(test_video_path), fourcc, 30.0, (640, 480))

        for i in range(30):  # 30 frames
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)

        out.release()

        # Run comparison
        print("\nRunning comparison (this may take a moment)...")
        result = comparison.benchmark_on_video(str(test_video_path), max_frames=30)

        print("\n" + "="*60)
        print(result.summary)
        print("="*60)

        # Generate report
        report = comparison.generate_comparison_report()
        report_path = Path("results/comparison_demo/report.md")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nFull report saved to: {report_path}")

        print("\nModel comparison demo complete!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demonstrations."""
    parser = argparse.ArgumentParser(description="Advanced Features Demonstration")
    parser.add_argument(
        "--demo",
        choices=["all", "mediapipe", "fusion", "water", "adaptive", "comparison"],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SwimVision Pro - Advanced Features Demo")
    print("="*60)

    demos = {
        "mediapipe": demo_mediapipe,
        "fusion": demo_model_fusion,
        "water": demo_water_surface_detection,
        "adaptive": demo_adaptive_tuning,
        "comparison": demo_model_comparison,
    }

    if args.demo == "all":
        for name, demo_func in demos.items():
            try:
                demo_func()
            except Exception as e:
                print(f"\nERROR in {name} demo: {e}")
                import traceback
                traceback.print_exc()
    else:
        demos[args.demo]()

    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
