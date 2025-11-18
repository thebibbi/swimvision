"""SwimVision Pro - Main Streamlit Application."""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

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
if "pose_estimator" not in st.session_state:
    st.session_state.pose_estimator = None
if "swimming_analyzer" not in st.session_state:
    st.session_state.swimming_analyzer = SwimmingKeypoints()
if "pose_overlay" not in st.session_state:
    st.session_state.pose_overlay = PoseOverlay()


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")

    # Mode selection
    mode = st.selectbox(
        "Mode",
        ["Upload Video", "Live Camera", "Compare"],
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

            camera.release()

        except Exception as e:
            st.error(f"Error loading video: {e}")

        # Process video button
        if st.button("🎬 Process Video", type="primary"):
            if st.session_state.pose_estimator is None:
                st.warning("Please initialize the model first (see sidebar)")
            else:
                process_video(temp_path, st.session_state.pose_estimator)

elif mode == "Live Camera":
    st.header("📸 Live Camera Analysis")

    camera_id = st.number_input(
        "Camera ID",
        min_value=0,
        max_value=10,
        value=0,
        help="Camera device ID (0 = default camera)",
    )

    if st.button("🎥 Start Live Analysis", type="primary"):
        if st.session_state.pose_estimator is None:
            st.warning("Please initialize the model first (see sidebar)")
        else:
            st.info("Live camera mode not yet implemented in this version")
            st.markdown("This feature will be available in the next update.")

elif mode == "Compare":
    st.header("⚖️ Compare Swimmers")
    st.info("Comparison mode will be available in Phase 2")


def process_video(video_path: Path, pose_estimator: YOLOPoseEstimator):
    """Process uploaded video and display results.

    Args:
        video_path: Path to video file.
        pose_estimator: YOLO pose estimator instance.
    """
    st.markdown("### Processing Video...")

    # Create placeholders
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        camera = VideoFileCamera(str(video_path))
        camera.open()

        total_frames = camera.get_total_frames()
        fps = camera.get_fps()

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

                # Draw visualizations
                if show_skeleton:
                    frame = st.session_state.pose_overlay.draw_skeleton(frame, pose_data)

                if show_bbox and pose_data.get("bbox"):
                    frame = st.session_state.pose_overlay.draw_bbox(frame, pose_data["bbox"])

                if show_angles:
                    angles = st.session_state.swimming_analyzer.get_body_angles(pose_data)
                    frame = st.session_state.pose_overlay.draw_angles(frame, pose_data, angles)

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

            # Limit processing rate (optional)
            time.sleep(max(0, 1.0 / fps - inference_time))

        camera.release()
        progress_bar.progress(1.0)
        st.success("✅ Video processing complete!")

    except Exception as e:
        st.error(f"Error processing video: {e}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>SwimVision Pro v0.1.0 - Phase 1: Core Infrastructure</p>
        <p>Built with Streamlit • YOLO11 • OpenCV</p>
    </div>
    """,
    unsafe_allow_html=True,
)
