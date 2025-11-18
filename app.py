"""SwimVision Pro - Main Streamlit Application with Phase 2 Analysis."""

import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.dtw_analyzer import DTWAnalyzer
from src.analysis.frechet_analyzer import FrechetAnalyzer
from src.analysis.stroke_phases import StrokePhaseDetector
from src.analysis.stroke_similarity import StrokeSimilarityEnsemble
from src.cameras.video_file import VideoFileCamera
from src.cameras.webcam import WebcamCamera
from src.pose.swimming_keypoints import SwimmingKeypoints
from src.pose.yolo_estimator import YOLOPoseEstimator
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
    if "analyzed_videos" not in st.session_state:
        st.session_state.analyzed_videos = {}


init_session_state()


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")

    # Mode selection
    mode = st.selectbox(
        "Mode",
        ["Upload Video", "Compare Strokes", "Stroke Analysis"],
        help="Select analysis mode",
    )

    st.markdown("---")

    # Pose estimation settings
    st.subheader("Pose Estimation")

    model_name = st.selectbox(
        "YOLO Model",
        ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"],
        index=0,
        help="Larger models are more accurate but slower",
    )

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
        with st.spinner("Loading YOLO model..."):
            try:
                st.session_state.pose_estimator = YOLOPoseEstimator(
                    model_name=model_name,
                    device=device,
                    confidence=confidence_threshold,
                )
                st.success("Model loaded successfully!")
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
                st.metric("Resolution", f"{camera.get_resolution()[0]}x{camera.get_resolution()[1]}")
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


def process_video(
    video_path: Path,
    pose_estimator: YOLOPoseEstimator,
    fps: float,
    analyze_phases: bool,
    extract_trajectory: bool,
) -> Dict:
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
    angles_over_time = {
        "left_elbow": [],
        "right_elbow": [],
        "left_shoulder": [],
        "right_shoulder": [],
    }

    try:
        camera = VideoFileCamera(str(video_path))
        camera.open()

        total_frames = camera.get_total_frames()

        # Initialize metrics
        frame_times = []
        detected_count = 0

        for frame_idx, frame in enumerate(camera.stream_frames()):
            # Update progress
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)

            # Estimate pose
            start_time = time.time()
            pose_data, _ = pose_estimator.estimate_pose(frame, return_image=False)
            inference_time = time.time() - start_time

            frame_times.append(inference_time)

            # Analyze pose
            if pose_data is not None:
                detected_count += 1
                pose_sequence.append(pose_data)

                # Extract trajectories
                if extract_trajectory:
                    left_wrist = pose_estimator.get_keypoint(pose_data, "left_wrist")
                    right_wrist = pose_estimator.get_keypoint(pose_data, "right_wrist")

                    if left_wrist:
                        left_hand_path.append((left_wrist[0], left_wrist[1]))
                    if right_wrist:
                        right_hand_path.append((right_wrist[0], right_wrist[1]))

                # Extract angles
                angles = st.session_state.swimming_analyzer.get_body_angles(pose_data)
                for angle_name in angles_over_time.keys():
                    value = angles.get(angle_name)
                    angles_over_time[angle_name].append(value if value is not None else np.nan)

                # Draw visualizations
                if show_skeleton:
                    frame = st.session_state.pose_overlay.draw_skeleton(frame, pose_data)

                if show_bbox and pose_data.get("bbox"):
                    frame = st.session_state.pose_overlay.draw_bbox(frame, pose_data["bbox"])

                if show_angles:
                    frame = st.session_state.pose_overlay.draw_angles(frame, pose_data, angles)

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
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

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
                phases = st.session_state.phase_detector.detect_phases_freestyle(
                    np.array(left_hand_path), fps, "left"
                )

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
        return None


def display_analysis_results(results: Dict, video_name: str):
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
            phase_df.append({
                "Phase": phase["phase"].value,
                "Start Frame": phase["start_frame"],
                "End Frame": phase["end_frame"],
                "Duration (s)": f"{phase['duration']:.2f}",
            })

        st.dataframe(phase_df, use_container_width=True)

        # Phase durations chart
        phase_durations = st.session_state.phase_detector.get_phase_durations(phases)

        if phase_durations:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(phase_durations.keys()),
                    y=list(phase_durations.values()),
                    marker_color="lightblue",
                )
            ])
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
            left_path = np.array(results["left_hand_path"])
            if len(left_path) > 0:
                fig = go.Figure(data=go.Scatter(
                    x=left_path[:, 0],
                    y=-left_path[:, 1],  # Invert Y for correct orientation
                    mode="lines+markers",
                    name="Left Hand",
                    line=dict(color="blue", width=2),
                ))
                fig.update_layout(
                    title="Left Hand Path",
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            right_path = np.array(results["right_hand_path"])
            if len(right_path) > 0:
                fig = go.Figure(data=go.Scatter(
                    x=right_path[:, 0],
                    y=-right_path[:, 1],
                    mode="lines+markers",
                    name="Right Hand",
                    line=dict(color="red", width=2),
                ))
                fig.update_layout(
                    title="Right Hand Path",
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)


def compare_strokes(video1_data: Dict, video2_data: Dict, fps: float) -> Dict:
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
    results = st.session_state.similarity_ensemble.comprehensive_comparison(
        stroke1, stroke2, fps
    )

    return results


def display_comparison_results(results: Dict, video1_name: str, video2_name: str):
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
        score_data.append({
            "Metric": metric.replace("_", " ").title(),
            "Score": f"{score:.1f}",
        })

    if score_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Radar chart
            categories = [s["Metric"] for s in score_data]
            values = [float(s["Score"]) for s in score_data]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Similarity Scores',
            ))

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
        for col, (measure, value) in zip(cols, additional.items()):
            with col:
                st.metric(
                    measure.replace("_", " ").title(),
                    f"{value:.2f}",
                )


def display_detailed_analysis(video_data: Dict, video_name: str):
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
                fig.add_trace(go.Scatter(
                    x=np.where(valid_indices)[0],
                    y=values_array[valid_indices],
                    mode="lines",
                    name=joint_name.replace("_", " ").title(),
                ))
                fig.update_layout(
                    title=joint_name.replace("_", " ").title(),
                    xaxis_title="Frame",
                    yaxis_title="Angle (degrees)",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>SwimVision Pro v0.2.0 - Phase 2: Time-Series Analysis</p>
        <p>Built with Streamlit • YOLO11 • OpenCV • DTW • Fréchet Distance</p>
    </div>
    """,
    unsafe_allow_html=True,
)
