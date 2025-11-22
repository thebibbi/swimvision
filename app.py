"""SwimVision Pro - Main Streamlit Application with Phase 2 Analysis."""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.dtw_analyzer import DTWAnalyzer
from src.analysis.features_extractor import FeaturesExtractor
from src.analysis.frechet_analyzer import FrechetAnalyzer
from src.analysis.stroke_phases import StrokePhaseDetector
from src.analysis.stroke_similarity import StrokeSimilarityEnsemble
from src.analysis.symmetry_analyzer import SymmetryAnalyzer
from src.cameras.video_file import VideoFileCamera
from src.pose.swimming_keypoints import SwimmingKeypoints
from src.pose.yolo_estimator import YOLOPoseEstimator

if TYPE_CHECKING:
    from src.pose.base_estimator import BasePoseEstimator
from src.visualization.pose_overlay import PoseOverlay

# Page configuration
st.set_page_config(
    page_title="SwimVision Pro",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🏊 SwimVision Pro")
st.markdown("**Real-time swimming technique analysis and injury prevention**")
st.markdown("---")


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "pose_estimator" not in st.session_state:
        st.session_state.pose_estimator = None
    if "swimming_analyzer" not in st.session_state:
        st.session_state.swimming_analyzer = SwimmingKeypoints()
    if "pose_overlay" not in st.session_state:
        st.session_state.pose_overlay = PoseOverlay()
    if "dtw_analyzer" not in st.session_state:
        st.session_state.dtw_analyzer = DTWAnalyzer()
    if "frechet_analyzer" not in st.session_state:
        st.session_state.frechet_analyzer = FrechetAnalyzer()
    if "phase_detector" not in st.session_state:
        st.session_state.phase_detector = StrokePhaseDetector()
    if "similarity_ensemble" not in st.session_state:
        st.session_state.similarity_ensemble = StrokeSimilarityEnsemble()
    if "features_extractor" not in st.session_state:
        st.session_state.features_extractor = FeaturesExtractor(fps=30.0)
    if "symmetry_analyzer" not in st.session_state:
        st.session_state.symmetry_analyzer = SymmetryAnalyzer(fps=30.0)
    if "analyzed_videos" not in st.session_state:
        st.session_state.analyzed_videos = {}


init_session_state()


# ============================================================================
# Helper Functions - Define before main content
# ============================================================================


AngleHistory = dict[str, list[float]]
VideoAnalysisResult = dict[str, Any]


def process_video(
    video_path: Path,
    pose_estimator: YOLOPoseEstimator,
    fps: float,
    analyze_phases: bool,
    extract_trajectory: bool,
) -> VideoAnalysisResult:
    """Process uploaded video and extract analysis data.

    Args:
        video_path: Path to video file.
        pose_estimator: YOLO pose estimator instance.
        fps: Video FPS.
        analyze_phases: Whether to detect stroke phases.
        extract_trajectory: Whether to extract hand trajectories.

    Returns:
        Dictionary with analysis results.
    """
    st.markdown("### Processing Video...")

    # Create placeholders
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Storage for analysis
    pose_sequence = []
    left_hand_path = []
    right_hand_path = []
    angles_over_time: AngleHistory = {
        "left_elbow": [],
        "right_elbow": [],
        "left_shoulder": [],
        "right_shoulder": [],
        "left_knee": [],
        "right_knee": [],
    }

    try:
        camera = VideoFileCamera(str(video_path))
        camera.open()

        total_frames = camera.get_total_frames()
        st.info(f"Processing {total_frames} frames from video")

        # Initialize metrics
        frame_times = []
        detected_count = 0

        # Get visualization settings from session state
        show_skeleton = st.session_state.get("show_skeleton", True)
        show_bbox = st.session_state.get("show_bbox", False)
        show_angles = st.session_state.get("show_angles", True)
        show_fps = st.session_state.get("show_fps", True)
        show_trajectory = st.session_state.get("show_trajectory", False)

        for frame_idx, frame in enumerate(camera.stream_frames()):
            # Update progress
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)

            # Estimate pose
            start_time = time.time()
            pose_result, _ = pose_estimator.estimate_pose(frame, return_image=False)
            inference_time = time.time() - start_time

            frame_times.append(inference_time)

            # Handle both single dict and list[dict] returns
            # Multi-person models (YOLO, RTMPose, ViTPose) return list[dict]
            # Single-person models (MediaPipe, SMPL-X) return single dict
            if pose_result is not None:
                # Normalize to list format
                if isinstance(pose_result, list):
                    pose_data_list = pose_result
                else:
                    pose_data_list = [pose_result]

                # For single-swimmer analysis, use first detection
                if len(pose_data_list) > 0:
                    pose_data = pose_data_list[0]
                    # Debug: Check what pose_data actually is
                    logger.debug(f"pose_data type: {type(pose_data)}")
                    if isinstance(pose_data, dict):
                        logger.debug(f"pose_data keys: {pose_data.keys()}")
                    else:
                        logger.error(f"ERROR: pose_data is not a dict! It's {type(pose_data)}")
                        # If it's a numpy array, we have a problem
                        if isinstance(pose_data, np.ndarray):
                            logger.error(f"pose_data shape: {pose_data.shape}")
                            # Skip this frame if pose_data is wrong type
                            continue

                    detected_count += 1
                    pose_sequence.append(pose_data)

                    # Extract trajectories
                    if extract_trajectory:
                        try:
                            logger.debug("Getting left wrist from pose_data")
                            left_wrist = pose_estimator.get_keypoint(pose_data, "left_wrist")
                            logger.debug("Getting right wrist from pose_data")
                            right_wrist = pose_estimator.get_keypoint(pose_data, "right_wrist")

                            if left_wrist:
                                left_hand_path.append((left_wrist[0], left_wrist[1]))
                            if right_wrist:
                                right_hand_path.append((right_wrist[0], right_wrist[1]))
                        except AttributeError as e:
                            import traceback

                            logger.error(f"Error in trajectory extraction: {e}")
                            logger.error(f"Full traceback: {traceback.format_exc()}")
                            logger.error(f"pose_data type: {type(pose_data)}")
                            if isinstance(pose_data, dict):
                                logger.error(f"pose_data keys: {pose_data.keys()}")
                                if "keypoints" in pose_data:
                                    logger.error(f"keypoints type: {type(pose_data['keypoints'])}")
                            else:
                                logger.error(f"pose_data content: {pose_data}")
                            raise

                    # Extract angles
                    try:
                        angles = st.session_state.swimming_analyzer.get_body_angles(pose_data)
                        for angle_name in angles_over_time:
                            value = angles.get(angle_name)
                            angles_over_time[angle_name].append(
                                float(value) if value is not None else float(np.nan)
                            )
                    except AttributeError as e:
                        logger.error(f"Error in get_body_angles: {e}")
                        logger.error(f"pose_data type: {type(pose_data)}")
                        if isinstance(pose_data, dict):
                            logger.error(f"pose_data keys: {pose_data.keys()}")
                        else:
                            logger.error(f"pose_data content: {pose_data}")
                        raise
                else:
                    pose_data = None
            else:
                pose_data = None
                # Debug: Log when pose is not detected
                if frame_idx % 30 == 0:  # Log every 30 frames to avoid spam
                    st.write(f"Frame {frame_idx}: No pose detected")

                # Debug: Check model and frame info
                if frame_idx == 0:
                    st.write(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    st.write(f"Model file exists: {Path('yolo11n-pose.pt').exists()}")
                    st.write(f"Pose estimator initialized: {pose_estimator.model is not None}")
                    st.write(f"Confidence threshold: {pose_estimator.confidence}")

            # Draw visualizations (only if pose was detected)
            if pose_data is not None:
                try:
                    if show_skeleton:
                        frame = st.session_state.pose_overlay.draw_skeleton(frame, pose_data)

                    if show_bbox and "bbox" in pose_data and pose_data["bbox"] is not None:
                        frame = st.session_state.pose_overlay.draw_bbox(frame, pose_data["bbox"])

                    if show_angles:
                        frame = st.session_state.pose_overlay.draw_angles(frame, pose_data, angles)
                except AttributeError as e:
                    logger.error(f"Error in pose overlay drawing: {e}")
                    logger.error(f"pose_data type: {type(pose_data)}")
                    if isinstance(pose_data, dict):
                        logger.error(f"pose_data keys: {pose_data.keys()}")
                    else:
                        logger.error(f"pose_data content: {pose_data}")
                    raise

                if show_trajectory and len(left_hand_path) > 1:
                    frame = st.session_state.pose_overlay.draw_trajectory(
                        frame, left_hand_path[-30:], color=(255, 255, 0)
                    )
                    frame = st.session_state.pose_overlay.draw_trajectory(
                        frame, right_hand_path[-30:], color=(0, 255, 255)
                    )

            # Draw FPS
            if show_fps:
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                frame = st.session_state.pose_overlay.draw_fps(frame, current_fps)

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB")

            # Display metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Frame", f"{frame_idx + 1}/{total_frames}")
                with col2:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    st.metric("Avg FPS", f"{avg_fps:.1f}")
                with col3:
                    st.metric("Detections", f"{detected_count}/{frame_idx + 1}")
                with col4:
                    detection_rate = (detected_count / (frame_idx + 1)) * 100
                    st.metric("Detection Rate", f"{detection_rate:.1f}%")

        camera.release()
        progress_bar.progress(1.0)
        st.success("✅ Video processing complete!")

        # Analyze phases if requested
        phases = None
        if analyze_phases and len(left_hand_path) > 10:
            with st.spinner("Detecting stroke phases..."):
                try:
                    phases = st.session_state.phase_detector.detect_phases_freestyle(
                        np.array(left_hand_path), fps, "left"
                    )
                except Exception as e:
                    st.warning(f"Could not detect stroke phases: {e}")
                    phases = None

        # Compile results
        return {
            "pose_sequence": pose_sequence,
            "left_hand_path": left_hand_path,
            "right_hand_path": right_hand_path,
            "angles": angles_over_time,
            "phases": phases,
            "fps": fps,
            "total_frames": total_frames,
            "detection_rate": detection_rate,
        }

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return {}


def display_analysis_results(results: dict, video_name: str):
    """Display analysis results for a single video.

    Args:
        results: Analysis results dictionary.
        video_name: Name of the video.
    """
    st.markdown("### Analysis Results")

    # Display phases if available
    if results.get("phases"):
        st.subheader("🔄 Detected Stroke Phases")

        phases = results["phases"]
        phase_df = []

        for phase in phases:
            phase_df.append(
                {
                    "Phase": phase["phase"].value,
                    "Start Frame": phase["start_frame"],
                    "End Frame": phase["end_frame"],
                    "Duration (s)": f"{phase['duration']:.2f}",
                }
            )

        st.dataframe(phase_df, use_container_width=True)

        # Phase durations chart
        phase_durations = st.session_state.phase_detector.get_phase_durations(phases)

        if phase_durations:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(phase_durations.keys()),
                        y=list(phase_durations.values()),
                        marker_color="lightblue",
                    )
                ]
            )
            fig.update_layout(
                title="Average Phase Durations",
                xaxis_title="Phase",
                yaxis_title="Duration (seconds)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Trajectory visualization
    if results.get("left_hand_path") and results.get("right_hand_path"):
        st.subheader("✋ Hand Trajectories")

        col1, col2 = st.columns(2)

        with col1:
            if results.get("left_hand_path") and len(results["left_hand_path"]) > 0:
                left_path = np.array(results["left_hand_path"])
                fig = go.Figure(
                    data=go.Scatter(
                        x=left_path[:, 0],
                        y=-left_path[:, 1],  # Invert Y for correct orientation
                        mode="lines+markers",
                        name="Left Hand",
                        line=dict(color="blue", width=2),
                    )
                )
                fig.update_layout(
                    title="Left Hand Path",
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No left hand trajectory data available")

        with col2:
            if results.get("right_hand_path") and len(results["right_hand_path"]) > 0:
                right_path = np.array(results["right_hand_path"])
                fig = go.Figure(
                    data=go.Scatter(
                        x=right_path[:, 0],
                        y=-right_path[:, 1],
                        mode="lines+markers",
                        name="Right Hand",
                        line=dict(color="red", width=2),
                    )
                )
                fig.update_layout(
                    title="Right Hand Path",
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No right hand trajectory data available")


def compare_strokes(video1_data: dict, video2_data: dict, fps: float) -> dict:
    """Compare two swimming strokes.

    Args:
        video1_data: Reference video data.
        video2_data: Comparison video data.
        fps: FPS for analysis.

    Returns:
        Comparison results dictionary.
    """
    # Prepare stroke data
    stroke1 = {
        "left_hand_path": video1_data["left_hand_path"],
        "right_hand_path": video1_data["right_hand_path"],
        "angles": {k: np.array(v) for k, v in video1_data["angles"].items()},
    }

    stroke2 = {
        "left_hand_path": video2_data["left_hand_path"],
        "right_hand_path": video2_data["right_hand_path"],
        "angles": {k: np.array(v) for k, v in video2_data["angles"].items()},
    }

    # Perform comprehensive comparison
    results = st.session_state.similarity_ensemble.comprehensive_comparison(stroke1, stroke2, fps)

    return results


def display_comparison_results(results: dict, video1_name: str, video2_name: str):
    """Display comparison results between two strokes.

    Args:
        results: Comparison results.
        video1_name: Reference video name.
        video2_name: Comparison video name.
    """
    st.markdown("### Comparison Results")

    # Overall score
    overall_score = results.get("overall_score", 0)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.metric(
            "Overall Similarity Score",
            f"{overall_score:.1f}/100",
            delta=None,
        )

    with col2:
        if overall_score >= 90:
            st.success("🌟 Excellent!")
        elif overall_score >= 75:
            st.info("👍 Good")
        elif overall_score >= 60:
            st.warning("⚠️ Needs Work")
        else:
            st.error("❌ Significant Differences")

    st.markdown("---")

    # Individual scores
    st.subheader("📊 Detailed Scores")

    scores = results.get("individual_scores", {})

    score_data = []
    for metric, score in scores.items():
        score_data.append(
            {
                "Metric": metric.replace("_", " ").title(),
                "Score": f"{score:.1f}",
            }
        )

    if score_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Radar chart
            categories = [s["Metric"] for s in score_data]
            values = [float(s["Score"]) for s in score_data]

            fig = go.Figure()

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill="toself",
                    name="Similarity Scores",
                )
            )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                    )
                ),
                showlegend=False,
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(score_data, use_container_width=True, hide_index=True)

    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")

    recommendations = results.get("recommendations", [])
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

    # Additional measures
    st.markdown("---")
    st.subheader("🔬 Additional Similarity Measures")

    additional = results.get("additional_measures", {})
    if additional:
        cols = st.columns(len(additional))
        for col, (measure, value) in zip(cols, additional.items(), strict=False):
            with col:
                st.metric(
                    measure.replace("_", " ").title(),
                    f"{value:.2f}",
                )


def display_detailed_analysis(video_data: dict, video_name: str):
    """Display detailed analysis for a single video.

    Args:
        video_data: Video analysis data.
        video_name: Name of the video.
    """
    st.markdown(f"### Analysis of: {video_name}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Frames", video_data["total_frames"])
    with col2:
        st.metric("FPS", f"{video_data['fps']:.1f}")
    with col3:
        st.metric("Detection Rate", f"{video_data['detection_rate']:.1f}%")
    with col4:
        duration = video_data["total_frames"] / video_data["fps"]
        st.metric("Duration", f"{duration:.1f}s")

    st.markdown("---")

    # Joint angles over time
    st.subheader("📐 Joint Angles Over Time")

    angles = video_data["angles"]

    for joint_name, values in angles.items():
        if len(values) > 0:
            values_array = np.array(values)
            # Remove NaN values for plotting
            valid_indices = ~np.isnan(values_array)

            if np.any(valid_indices):
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=np.where(valid_indices)[0],
                        y=values_array[valid_indices],
                        mode="lines",
                        name=joint_name.replace("_", " ").title(),
                    )
                )
                fig.update_layout(
                    title=joint_name.replace("_", " ").title(),
                    xaxis_title="Frame",
                    yaxis_title="Angle (degrees)",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Sidebar Configuration
# ============================================================================

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")

    # Mode selection
    mode = st.selectbox(
        "Mode",
        [
            "Upload Video",
            "Compare Strokes",
            "Stroke Analysis",
            "Biomechanical Features",
            "Symmetry Analysis",
        ],
        help="Select analysis mode",
    )

    st.markdown("---")

    # Pose estimation settings
    st.subheader("Pose Estimation")

    pose_models = [
        "YOLO11",
        "MediaPipe",
        "OpenPose",
        "AlphaPose",
        "RTMPose",
        "ViTPose",
        "SMPL-X",
        "FreeMoCap (Multi-Camera 3D)",
        "Multi-Model Fusion",
    ]

    pose_model_type = st.selectbox(
        "Pose Model",
        pose_models,
        index=0,
        help="Select pose estimation model",
    )

    # Model-specific options
    if pose_model_type == "YOLO11":
        model_variant = st.selectbox(
            "YOLO Variant",
            ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"],
            index=0,
            help="Larger models are more accurate but slower",
        )
    elif pose_model_type == "MediaPipe":
        model_variant = st.selectbox(
            "MediaPipe Complexity",
            ["Lite (0)", "Full (1)", "Heavy (2)"],
            index=1,
            help="Model complexity level",
        )
    elif pose_model_type == "AlphaPose":
        model_variant = st.selectbox(
            "AlphaPose Model",
            ["halpe26", "coco", "coco_wholebody"],
            index=0,
            help="AlphaPose model variant",
        )
    elif pose_model_type == "RTMPose":
        model_variant = st.selectbox(
            "RTMPose Variant",
            ["rtmpose-t", "rtmpose-s", "rtmpose-m", "rtmpose-l"],
            index=2,
            help="MMPose RTMPose variants (tiny → large)",
        )
    elif pose_model_type == "ViTPose":
        model_variant = st.selectbox(
            "ViTPose Variant",
            ["vitpose-b", "vitpose-l", "vitpose-h"],
            index=0,
            help="ViTPose variants (base/large/huge)",
        )
    elif pose_model_type == "FreeMoCap (Multi-Camera 3D)":
        st.info("🎥 FreeMoCap enables multi-camera 3D motion capture")

        camera_count = st.number_input(
            "Number of Cameras",
            min_value=1,
            max_value=8,
            value=4,
            help="Number of synchronized cameras for 3D reconstruction",
        )

        calibration_file = st.file_uploader(
            "Upload Calibration File",
            type=["toml", "json"],
            help="Camera calibration from FreeMoCap or Anipose",
        )

        use_charuco = st.checkbox(
            "Use CharuCo Board Calibration",
            value=False,
            help="Use CharuCo board for automatic calibration",
        )

        if calibration_file:
            st.success(f"✅ Calibration file loaded: {calibration_file.name}")
        else:
            st.warning("⚠️ No calibration file provided. Single-camera mode will be used.")

        model_variant = {
            "camera_count": camera_count,
            "calibration_file": calibration_file,
            "use_charuco": use_charuco,
        }

    elif pose_model_type == "Multi-Model Fusion":
        fusion_method = st.selectbox(
            "Fusion Method",
            ["Weighted Average", "Median", "Max Confidence", "Kalman Fusion"],
            help="Method for fusing predictions",
        )
        st.multiselect(
            "Models to Fuse",
            ["YOLO11", "MediaPipe", "OpenPose", "RTMPose", "ViTPose"],
            default=["YOLO11", "MediaPipe"],
            help="Select models to combine",
        )
    else:
        model_variant = None

    device = st.selectbox(
        "Device",
        ["cpu", "cuda", "mps"],
        index=0,
        help="Use GPU (cuda/mps) if available",
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for pose detection",
    )

    # Advanced Features
    with st.expander("🚀 Advanced Features"):
        enable_water_surface = st.checkbox(
            "Water Surface Detection",
            value=False,
            help="Detect water surface for entry/exit events",
        )

        enable_adaptive_tuning = st.checkbox(
            "Adaptive Threshold Tuning",
            value=False,
            help="Auto-tune thresholds based on conditions",
        )

        if enable_water_surface:
            water_detection_method = st.selectbox(
                "Surface Detection Method",
                ["Edge Detection", "Color Segmentation", "Hybrid"],
                index=2,
                help="Method for detecting water surface",
            )

        st.session_state.enable_water_surface = enable_water_surface
        st.session_state.enable_adaptive_tuning = enable_adaptive_tuning

    st.markdown("---")

    # Occlusion Tracking settings
    st.subheader("🔍 Occlusion Tracking")

    enable_occlusion_tracking = st.checkbox(
        "Enable Occlusion Tracking", value=False, help="Handle underwater hand tracking"
    )

    if enable_occlusion_tracking:
        tracking_method = st.selectbox(
            "Tracking Method",
            [
                "Hybrid (Recommended)",
                "Kalman Prediction",
                "Phase-Aware",
                "Interpolation",
                "Kalman Only",
            ],
            help="Method for handling occluded hands",
        )

        # Map to enum values
        tracking_method_map = {
            "Hybrid (Recommended)": "hybrid",
            "Kalman Prediction": "kalman_predict",
            "Phase-Aware": "phase_aware",
            "Interpolation": "interpolation",
            "Kalman Only": "kalman",
        }
        st.session_state.tracking_method = tracking_method_map[tracking_method]

        show_occlusion_overlay = st.checkbox("Show Occlusion Overlay", value=True)
        show_prediction_confidence = st.checkbox("Show Prediction Confidence", value=True)
    else:
        st.session_state.tracking_method = None
        show_occlusion_overlay = False
        show_prediction_confidence = False

    # Store in session state for access in processing
    st.session_state.enable_occlusion_tracking = enable_occlusion_tracking
    st.session_state.show_occlusion_overlay = show_occlusion_overlay
    st.session_state.show_prediction_confidence = show_prediction_confidence

    st.markdown("---")

    # Visualization settings
    st.subheader("Visualization")

    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_angles = st.checkbox("Show Angles", value=True)
    show_bbox = st.checkbox("Show Bounding Box", value=False)
    show_fps = st.checkbox("Show FPS", value=True)
    show_trajectory = st.checkbox("Show Hand Trajectory", value=False)

    st.markdown("---")

    # Model initialization
    if st.button("Initialize/Reload Model", type="primary"):
        with st.spinner(f"Loading {pose_model_type} model..."):
            try:
                if pose_model_type == "YOLO11":
                    from src.pose.yolo_estimator import YOLOPoseEstimator

                    st.session_state.pose_estimator = YOLOPoseEstimator(
                        model_name=model_variant,
                        device=device,
                        confidence=confidence_threshold,
                    )

                elif pose_model_type == "MediaPipe":
                    from src.pose.mediapipe_estimator import MediaPipeEstimator

                    complexity_map = {"Lite (0)": 0, "Full (1)": 1, "Heavy (2)": 2}
                    complexity = complexity_map[model_variant]
                    st.session_state.pose_estimator = MediaPipeEstimator(
                        model_complexity=complexity,
                        min_detection_confidence=confidence_threshold,
                        device=device,
                    )

                elif pose_model_type == "OpenPose":
                    from src.pose.openpose_estimator import OpenPoseEstimator

                    st.session_state.pose_estimator = OpenPoseEstimator(
                        device=device,
                        confidence=confidence_threshold,
                    )

                elif pose_model_type == "AlphaPose":
                    from src.pose.alphapose_estimator import AlphaPoseEstimator

                    st.session_state.pose_estimator = AlphaPoseEstimator(
                        model_name=model_variant,
                        device=device,
                        confidence=confidence_threshold,
                    )

                elif pose_model_type == "RTMPose":
                    from src.pose.rtmpose_estimator import RTMPoseEstimator

                    st.session_state.pose_estimator = RTMPoseEstimator(
                        model_variant=model_variant,
                        device=device,
                        confidence=confidence_threshold,
                    )

                elif pose_model_type == "ViTPose":
                    from src.pose.vitpose_estimator import ViTPoseEstimator

                    st.session_state.pose_estimator = ViTPoseEstimator(
                        model_variant=model_variant,
                        device=device,
                        confidence=confidence_threshold,
                    )

                elif pose_model_type == "SMPL-X":
                    from src.pose.smpl_estimator import SMPLEstimator

                    st.session_state.pose_estimator = SMPLEstimator(
                        model_type="smplx",
                        device=device,
                        confidence=confidence_threshold,
                    )

                elif pose_model_type == "Multi-Model Fusion":
                    from src.pose.mediapipe_estimator import MediaPipeEstimator
                    from src.pose.model_fusion import FusionMethod, MultiModelFusion
                    from src.pose.rtmpose_estimator import RTMPoseEstimator
                    from src.pose.vitpose_estimator import ViTPoseEstimator
                    from src.pose.yolo_estimator import YOLOPoseEstimator

                    # Create individual models
                    models: list["BasePoseEstimator"] = []
                    models.append(
                        YOLOPoseEstimator("yolo11n-pose.pt", device, confidence_threshold)
                    )
                    models.append(MediaPipeEstimator(1, confidence_threshold, device))
                    models.append(
                        RTMPoseEstimator(
                            "rtmpose-m", device=device, confidence=confidence_threshold
                        )
                    )
                    models.append(
                        ViTPoseEstimator(
                            "vitpose-b", device=device, confidence=confidence_threshold
                        )
                    )

                    # Map fusion method
                    method_map = {
                        "Weighted Average": FusionMethod.WEIGHTED_AVERAGE,
                        "Median": FusionMethod.MEDIAN,
                        "Max Confidence": FusionMethod.MAX_CONFIDENCE,
                        "Kalman Fusion": FusionMethod.KALMAN_FUSION,
                    }

                    st.session_state.pose_estimator = MultiModelFusion(
                        models=models,
                        fusion_method=method_map[fusion_method],
                        confidence_threshold=confidence_threshold,
                    )

                st.success(f"{pose_model_type} model loaded successfully!")

                # Initialize water surface detector if enabled
                if st.session_state.get("enable_water_surface", False):
                    from src.analysis.water_surface_detector import WaterSurfaceDetector

                    water_detection_map = {
                        "Edge Detection": "edge",
                        "Color Segmentation": "color",
                        "Hybrid": "hybrid",
                    }
                    st.session_state.water_surface_detector = WaterSurfaceDetector(
                        detection_method=water_detection_map.get(water_detection_method, "hybrid"),
                    )

                # Initialize adaptive tuner if enabled
                if st.session_state.get("enable_adaptive_tuning", False):
                    from src.utils.adaptive_tuning import AdaptiveThresholdTuner

                    st.session_state.adaptive_tuner = AdaptiveThresholdTuner(auto_tune=True)

            except ImportError as e:
                st.error(f"Model not available. Please install required dependencies: {e}")
            except Exception as e:
                st.error(f"Failed to load model: {e}")


# Main content area
if mode == "Upload Video":
    st.header("📹 Upload Video Analysis")

    uploaded_file = st.file_uploader(
        "Choose a swimming video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video file for analysis",
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("data/videos") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded: {uploaded_file.name}")

        # Video info
        col1, col2, col3 = st.columns(3)

        try:
            camera = VideoFileCamera(str(temp_path))
            camera.open()

            with col1:
                st.metric(
                    "Resolution", f"{camera.get_resolution()[0]}x{camera.get_resolution()[1]}"
                )
            with col2:
                st.metric("FPS", f"{camera.get_fps():.1f}")
            with col3:
                st.metric("Duration", f"{camera.get_duration_seconds():.1f}s")

            fps = camera.get_fps()
            camera.release()

        except Exception as e:
            st.error(f"Error loading video: {e}")
            fps = 30.0

        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            analyze_phases = st.checkbox("Detect Stroke Phases", value=True)
        with col2:
            extract_trajectory = st.checkbox("Extract Hand Trajectories", value=True)

        # Process video button
        if st.button("🎬 Process Video", type="primary"):
            if st.session_state.pose_estimator is None:
                st.warning("Please initialize the model first (see sidebar)")
            else:
                analysis_results = process_video(
                    temp_path,
                    st.session_state.pose_estimator,
                    fps,
                    analyze_phases,
                    extract_trajectory,
                )

                # Store results
                if analysis_results:
                    st.session_state.analyzed_videos[uploaded_file.name] = analysis_results
                    display_analysis_results(analysis_results, uploaded_file.name)

elif mode == "Compare Strokes":
    st.header("⚖️ Compare Swimming Strokes")

    if len(st.session_state.analyzed_videos) < 2:
        st.info("Please upload and analyze at least 2 videos first using 'Upload Video' mode.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            video1_name = st.selectbox(
                "Reference Video (Ideal Technique)",
                options=list(st.session_state.analyzed_videos.keys()),
                key="video1",
            )

        with col2:
            video2_name = st.selectbox(
                "Comparison Video (Your Stroke)",
                options=[k for k in st.session_state.analyzed_videos.keys() if k != video1_name],
                key="video2",
            )

        if st.button("🔍 Compare Strokes", type="primary"):
            with st.spinner("Analyzing stroke similarity..."):
                video1_data = st.session_state.analyzed_videos[video1_name]
                video2_data = st.session_state.analyzed_videos[video2_name]

                comparison_results = compare_strokes(video1_data, video2_data, video1_data["fps"])

                display_comparison_results(comparison_results, video1_name, video2_name)

elif mode == "Stroke Analysis":
    st.header("📊 Detailed Stroke Analysis")

    if len(st.session_state.analyzed_videos) == 0:
        st.info("Please upload and analyze a video first using 'Upload Video' mode.")
    else:
        selected_video = st.selectbox(
            "Select Video to Analyze",
            options=list(st.session_state.analyzed_videos.keys()),
        )

        video_data = st.session_state.analyzed_videos[selected_video]

        # Display detailed analysis
        display_detailed_analysis(video_data, selected_video)

elif mode == "Biomechanical Features":
    st.header("🔬 Biomechanical Feature Analysis")

    if len(st.session_state.analyzed_videos) == 0:
        st.info("Please upload and analyze a video first using 'Upload Video' mode.")
    else:
        selected_video = st.selectbox(
            "Select Video to Analyze",
            options=list(st.session_state.analyzed_videos.keys()),
        )

        video_data = st.session_state.analyzed_videos[selected_video]

        # Update FPS for extractors if needed
        fps = video_data.get("fps", 30.0)
        st.session_state.features_extractor.fps = fps
        st.session_state.features_extractor.dt = 1.0 / fps

        # Extract features
        with st.spinner("Extracting biomechanical features..."):
            left_hand_path = (
                np.array(video_data["left_hand_path"]) if video_data["left_hand_path"] else None
            )
            right_hand_path = (
                np.array(video_data["right_hand_path"]) if video_data["right_hand_path"] else None
            )
            angles_over_time = {k: np.array(v) for k, v in video_data["angles"].items()}

            features = st.session_state.features_extractor.extract_all_features(
                video_data["pose_sequence"],
                left_hand_path,
                right_hand_path,
                angles_over_time,
            )

        # Display features
        st.subheader(f"Extracted Features for: {selected_video}")

        # Group features by category
        temporal_features = {
            k: v
            for k, v in features.items()
            if any(x in k for x in ["stroke_rate", "cycle", "tempo", "num_strokes"])
        }
        kinematic_features = {
            k: v
            for k, v in features.items()
            if any(
                x in k for x in ["velocity", "acceleration", "path", "displacement", "efficiency"]
            )
        }
        angular_features = {
            k: v
            for k, v in features.items()
            if any(x in k for x in ["elbow", "shoulder"])
            and not any(x in k for x in ["drop", "extreme", "asymmetry"])
        }
        symmetry_features = {k: v for k, v in features.items() if "asymmetry" in k}
        injury_features = {
            k: v
            for k, v in features.items()
            if any(x in k for x in ["extreme", "drop", "risk", "workload"])
        }

        # Temporal Features
        if temporal_features:
            st.markdown("### ⏱️ Temporal Features")
            cols = st.columns(min(len(temporal_features), 4))
            for idx, (key, value) in enumerate(temporal_features.items()):
                with cols[idx % 4]:
                    st.metric(key.replace("_", " ").title(), f"{value:.2f}")

        st.markdown("---")

        # Kinematic Features
        if kinematic_features:
            st.markdown("### 🚀 Kinematic Features")
            df_data = []
            for key, value in kinematic_features.items():
                df_data.append({"Feature": key.replace("_", " ").title(), "Value": f"{value:.2f}"})
            st.dataframe(df_data, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Angular Features
        if angular_features:
            st.markdown("### 📐 Angular Features")
            df_data = []
            for key, value in angular_features.items():
                df_data.append({"Feature": key.replace("_", " ").title(), "Value": f"{value:.2f}°"})
            st.dataframe(df_data, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Symmetry Features
        if symmetry_features:
            st.markdown("### ⚖️ Symmetry Features")
            cols = st.columns(min(len(symmetry_features), 3))
            for idx, (key, value) in enumerate(symmetry_features.items()):
                with cols[idx % 3]:
                    # Add color coding based on asymmetry level
                    if value < 10:
                        st.metric(
                            key.replace("_", " ").title(),
                            f"{value:.2f}%",
                            delta="Good",
                            delta_color="normal",
                        )
                    elif value < 20:
                        st.metric(
                            key.replace("_", " ").title(),
                            f"{value:.2f}%",
                            delta="Warning",
                            delta_color="off",
                        )
                    else:
                        st.metric(
                            key.replace("_", " ").title(),
                            f"{value:.2f}%",
                            delta="High",
                            delta_color="inverse",
                        )

        st.markdown("---")

        # Injury Risk Features
        if injury_features:
            st.markdown("### ⚠️ Injury Risk Indicators")
            df_data = []
            for key, value in injury_features.items():
                df_data.append(
                    {"Indicator": key.replace("_", " ").title(), "Value": f"{value:.2f}"}
                )
            st.dataframe(df_data, use_container_width=True, hide_index=True)

        # Feature export
        st.markdown("---")
        st.subheader("💾 Export Features")

        import json

        features_json = json.dumps(features, indent=2)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Features (JSON)",
                data=features_json,
                file_name=f"{selected_video}_features.json",
                mime="application/json",
            )

elif mode == "Symmetry Analysis":
    st.header("⚖️ Symmetry Analysis")

    if len(st.session_state.analyzed_videos) == 0:
        st.info("Please upload and analyze a video first using 'Upload Video' mode.")
    else:
        selected_video = st.selectbox(
            "Select Video to Analyze",
            options=list(st.session_state.analyzed_videos.keys()),
        )

        video_data = st.session_state.analyzed_videos[selected_video]

        # Update FPS for analyzers
        fps = video_data.get("fps", 30.0)
        st.session_state.symmetry_analyzer.fps = fps
        st.session_state.symmetry_analyzer.dt = 1.0 / fps

        # Perform symmetry analysis
        with st.spinner("Analyzing symmetry..."):
            left_hand_path = (
                np.array(video_data["left_hand_path"]) if video_data["left_hand_path"] else None
            )
            right_hand_path = (
                np.array(video_data["right_hand_path"]) if video_data["right_hand_path"] else None
            )
            angles_over_time = {k: np.array(v) for k, v in video_data["angles"].items()}

            symmetry_results = st.session_state.symmetry_analyzer.comprehensive_symmetry_analysis(
                left_hand_path,
                right_hand_path,
                angles_over_time,
            )

        # Display overall score
        st.subheader(f"Symmetry Analysis: {selected_video}")

        overall_score = symmetry_results["overall_symmetry_score"]
        interpretation = symmetry_results["interpretation"]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.metric("Overall Symmetry Score", f"{overall_score:.1f}/100")

        with col2:
            if overall_score >= 90:
                st.success("🌟 Excellent!")
            elif overall_score >= 75:
                st.info("👍 Good")
            elif overall_score >= 60:
                st.warning("⚠️ Needs Work")
            else:
                st.error("❌ Poor Symmetry")

        st.markdown(f"**{interpretation}**")

        st.markdown("---")

        # Arm Symmetry
        st.subheader("💪 Arm Symmetry")
        arm_symmetry = symmetry_results["arm_symmetry"]

        col1, col2, col3 = st.columns(3)

        with col1:
            if "path_length_asymmetry_pct" in arm_symmetry:
                st.metric(
                    "Path Length Asymmetry", f"{arm_symmetry['path_length_asymmetry_pct']:.1f}%"
                )
            if "left_path_length" in arm_symmetry:
                st.metric("Left Path Length", f"{arm_symmetry['left_path_length']:.1f} px")

        with col2:
            if "speed_asymmetry_pct" in arm_symmetry:
                st.metric("Speed Asymmetry", f"{arm_symmetry['speed_asymmetry_pct']:.1f}%")
            if "right_path_length" in arm_symmetry:
                st.metric("Right Path Length", f"{arm_symmetry['right_path_length']:.1f} px")

        with col3:
            if "elbow_angle_asymmetry_mean" in arm_symmetry:
                st.metric(
                    "Elbow Angle Asymmetry", f"{arm_symmetry['elbow_angle_asymmetry_mean']:.1f}°"
                )

        # Speed comparison chart
        if "left_mean_speed" in arm_symmetry and "right_mean_speed" in arm_symmetry:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Left Arm", "Right Arm"],
                        y=[arm_symmetry["left_mean_speed"], arm_symmetry["right_mean_speed"]],
                        marker_color=["blue", "red"],
                    )
                ]
            )
            fig.update_layout(
                title="Mean Speed Comparison",
                yaxis_title="Speed (pixels/frame)",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Temporal Symmetry
        st.subheader("⏱️ Temporal Symmetry")
        temporal_symmetry = symmetry_results["temporal_symmetry"]

        col1, col2, col3 = st.columns(3)

        with col1:
            if "left_num_strokes" in temporal_symmetry:
                st.metric("Left Strokes", f"{int(temporal_symmetry['left_num_strokes'])}")
            if "left_stroke_consistency" in temporal_symmetry:
                st.metric("Left Consistency", f"{temporal_symmetry['left_stroke_consistency']:.2f}")

        with col2:
            if "right_num_strokes" in temporal_symmetry:
                st.metric("Right Strokes", f"{int(temporal_symmetry['right_num_strokes'])}")
            if "right_stroke_consistency" in temporal_symmetry:
                st.metric(
                    "Right Consistency", f"{temporal_symmetry['right_stroke_consistency']:.2f}"
                )

        with col3:
            if "stroke_count_asymmetry_pct" in temporal_symmetry:
                st.metric(
                    "Stroke Count Asymmetry",
                    f"{temporal_symmetry['stroke_count_asymmetry_pct']:.1f}%",
                )

        st.markdown("---")

        # Force Imbalance
        st.subheader("⚡ Force Imbalance Estimate")
        force_imbalance = symmetry_results["force_imbalance"]

        col1, col2, col3 = st.columns(3)

        with col1:
            if "left_mean_acceleration" in force_imbalance:
                st.metric(
                    "Left Mean Acceleration", f"{force_imbalance['left_mean_acceleration']:.2f}"
                )

        with col2:
            if "right_mean_acceleration" in force_imbalance:
                st.metric(
                    "Right Mean Acceleration", f"{force_imbalance['right_mean_acceleration']:.2f}"
                )

        with col3:
            if "force_imbalance_pct" in force_imbalance:
                st.metric("Force Imbalance", f"{force_imbalance['force_imbalance_pct']:.1f}%")

        st.markdown("---")

        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = symmetry_results["recommendations"]

        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>SwimVision Pro v0.3.0 - Phase 3: Biomechanical Feature Extraction</p>
        <p>Built with Streamlit • YOLO11 • OpenCV • DTW • Kalman Filtering • Symmetry Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)
