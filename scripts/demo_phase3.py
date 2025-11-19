"""Phase 3 Demonstration: Biomechanical Feature Extraction & Symmetry Analysis.

This script demonstrates the Phase 3 capabilities:
- 30+ biomechanical features extraction
- Kalman filtering for trajectory smoothing
- Symmetry analysis (arm, temporal, force balance)
- Feature categorization and interpretation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.analysis.features_extractor import FeaturesExtractor, format_features_table
from src.analysis.symmetry_analyzer import SymmetryAnalyzer
from src.utils.smoothing import (
    smooth_trajectory_kalman,
    smooth_trajectory_savgol,
    smooth_trajectory_ma,
    calculate_velocity,
    calculate_acceleration,
    calculate_speed,
)


def generate_synthetic_swimming_data(num_frames=100, fps=30.0):
    """Generate synthetic swimming pose data for demonstration.

    Args:
        num_frames: Number of frames to generate.
        fps: Frame rate.

    Returns:
        Dictionary with pose data.
    """
    t = np.linspace(0, num_frames / fps, num_frames)

    # Generate left hand path (sinusoidal pattern with noise)
    left_hand_path = np.zeros((num_frames, 2))
    left_hand_path[:, 0] = 100 + 50 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 2, num_frames)
    left_hand_path[:, 1] = 200 + 30 * np.cos(2 * np.pi * 0.5 * t) + np.random.normal(0, 2, num_frames)

    # Generate right hand path (similar but with slight asymmetry)
    right_hand_path = np.zeros((num_frames, 2))
    right_hand_path[:, 0] = 300 + 45 * np.sin(2 * np.pi * 0.5 * t + 0.2) + np.random.normal(0, 2, num_frames)
    right_hand_path[:, 1] = 200 + 28 * np.cos(2 * np.pi * 0.5 * t + 0.2) + np.random.normal(0, 2, num_frames)

    # Generate joint angles
    left_elbow = 90 + 40 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 5, num_frames)
    right_elbow = 92 + 38 * np.sin(2 * np.pi * 0.5 * t + 0.2) + np.random.normal(0, 5, num_frames)
    left_shoulder = 120 + 30 * np.cos(2 * np.pi * 0.5 * t) + np.random.normal(0, 3, num_frames)
    right_shoulder = 118 + 32 * np.cos(2 * np.pi * 0.5 * t + 0.2) + np.random.normal(0, 3, num_frames)

    # Clip angles to valid ranges
    left_elbow = np.clip(left_elbow, 30, 180)
    right_elbow = np.clip(right_elbow, 30, 180)
    left_shoulder = np.clip(left_shoulder, 60, 180)
    right_shoulder = np.clip(right_shoulder, 60, 180)

    angles_over_time = {
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
    }

    # Create placeholder pose sequence
    pose_sequence = [{"frame": i} for i in range(num_frames)]

    return {
        "pose_sequence": pose_sequence,
        "left_hand_path": left_hand_path,
        "right_hand_path": right_hand_path,
        "angles_over_time": angles_over_time,
        "fps": fps,
    }


def demo_smoothing():
    """Demonstrate trajectory smoothing techniques."""
    print("\n" + "=" * 80)
    print("  DEMONSTRATION 1: Trajectory Smoothing")
    print("=" * 80)

    # Generate noisy trajectory
    t = np.linspace(0, 2 * np.pi, 50)
    true_trajectory = np.column_stack([
        100 + 50 * np.sin(t),
        200 + 30 * np.cos(t)
    ])

    # Add noise
    noisy_trajectory = true_trajectory + np.random.normal(0, 5, true_trajectory.shape)

    # Apply different smoothing techniques
    kalman_smoothed, velocities = smooth_trajectory_kalman(noisy_trajectory)
    savgol_smoothed = smooth_trajectory_savgol(noisy_trajectory, window_length=11, polyorder=3)
    ma_smoothed = smooth_trajectory_ma(noisy_trajectory, window_size=5)

    print(f"\nOriginal trajectory: {len(noisy_trajectory)} points")
    print(f"RMS noise (Kalman): {np.sqrt(np.mean((kalman_smoothed - true_trajectory)**2)):.2f} pixels")
    print(f"RMS noise (Savitzky-Golay): {np.sqrt(np.mean((savgol_smoothed - true_trajectory)**2)):.2f} pixels")
    print(f"RMS noise (Moving Average): {np.sqrt(np.mean((ma_smoothed - true_trajectory)**2)):.2f} pixels")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Noisy vs True
    axes[0, 0].plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], 'o', alpha=0.5, label='Noisy', markersize=4)
    axes[0, 0].plot(true_trajectory[:, 0], true_trajectory[:, 1], 'r-', linewidth=2, label='True')
    axes[0, 0].set_title('Noisy vs True Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Kalman filtering
    axes[0, 1].plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], 'o', alpha=0.3, label='Noisy', markersize=4)
    axes[0, 1].plot(kalman_smoothed[:, 0], kalman_smoothed[:, 1], 'g-', linewidth=2, label='Kalman')
    axes[0, 1].plot(true_trajectory[:, 0], true_trajectory[:, 1], 'r--', linewidth=1, label='True')
    axes[0, 1].set_title('Kalman Filtering')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Savitzky-Golay filtering
    axes[1, 0].plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], 'o', alpha=0.3, label='Noisy', markersize=4)
    axes[1, 0].plot(savgol_smoothed[:, 0], savgol_smoothed[:, 1], 'b-', linewidth=2, label='Savitzky-Golay')
    axes[1, 0].plot(true_trajectory[:, 0], true_trajectory[:, 1], 'r--', linewidth=1, label='True')
    axes[1, 0].set_title('Savitzky-Golay Filtering')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Moving average
    axes[1, 1].plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], 'o', alpha=0.3, label='Noisy', markersize=4)
    axes[1, 1].plot(ma_smoothed[:, 0], ma_smoothed[:, 1], 'm-', linewidth=2, label='Moving Average')
    axes[1, 1].plot(true_trajectory[:, 0], true_trajectory[:, 1], 'r--', linewidth=1, label='True')
    axes[1, 1].set_title('Moving Average Filtering')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path("data/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "phase3_smoothing_demo.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Smoothing visualization saved to: {output_dir / 'phase3_smoothing_demo.png'}")

    plt.close()


def demo_feature_extraction():
    """Demonstrate biomechanical feature extraction."""
    print("\n" + "=" * 80)
    print("  DEMONSTRATION 2: Biomechanical Feature Extraction")
    print("=" * 80)

    # Generate synthetic data
    data = generate_synthetic_swimming_data(num_frames=150, fps=30.0)

    # Create extractor
    extractor = FeaturesExtractor(fps=data["fps"])

    # Extract all features
    print("\n⏳ Extracting features...")
    features = extractor.extract_all_features(
        data["pose_sequence"],
        data["left_hand_path"],
        data["right_hand_path"],
        data["angles_over_time"],
    )

    print(f"✅ Extracted {len(features)} features!")

    # Print feature table
    print("\n" + format_features_table(features))

    # Key insights
    print("\n" + "=" * 80)
    print("  Key Insights")
    print("=" * 80)

    if "stroke_rate" in features:
        print(f"\n📊 Stroke Rate: {features['stroke_rate']:.1f} strokes/minute")

    if "stroke_cycle_time" in features:
        print(f"⏱️  Cycle Time: {features['stroke_cycle_time']:.2f} seconds")

    if "left_hand_mean_velocity" in features:
        print(f"🚀 Left Hand Mean Velocity: {features['left_hand_mean_velocity']:.2f} pixels/frame")

    if "right_hand_mean_velocity" in features:
        print(f"🚀 Right Hand Mean Velocity: {features['right_hand_mean_velocity']:.2f} pixels/frame")

    if "path_length_asymmetry" in features:
        asym = features['path_length_asymmetry']
        print(f"\n⚖️  Path Length Asymmetry: {asym:.1f}%")
        if asym < 10:
            print("   → Excellent symmetry!")
        elif asym < 20:
            print("   → Good symmetry")
        else:
            print("   → Significant asymmetry detected")

    # Visualize key features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Hand trajectories
    axes[0, 0].plot(data["left_hand_path"][:, 0], data["left_hand_path"][:, 1], 'b-', linewidth=2, label='Left Hand')
    axes[0, 0].plot(data["right_hand_path"][:, 0], data["right_hand_path"][:, 1], 'r-', linewidth=2, label='Right Hand')
    axes[0, 0].set_title('Hand Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('X Position (pixels)')
    axes[0, 0].set_ylabel('Y Position (pixels)')

    # Plot 2: Elbow angles
    axes[0, 1].plot(data["angles_over_time"]["left_elbow"], 'b-', linewidth=2, label='Left Elbow')
    axes[0, 1].plot(data["angles_over_time"]["right_elbow"], 'r-', linewidth=2, label='Right Elbow')
    axes[0, 1].set_title('Elbow Angles Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Angle (degrees)')

    # Plot 3: Velocities
    left_velocities = calculate_velocity(data["left_hand_path"], dt=1.0/data["fps"])
    right_velocities = calculate_velocity(data["right_hand_path"], dt=1.0/data["fps"])

    if len(left_velocities) > 0:
        left_speeds = calculate_speed(left_velocities)
        axes[1, 0].plot(left_speeds, 'b-', linewidth=2, label='Left Hand')

    if len(right_velocities) > 0:
        right_speeds = calculate_speed(right_velocities)
        axes[1, 0].plot(right_speeds, 'r-', linewidth=2, label='Right Hand')

    axes[1, 0].set_title('Hand Speed Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Speed (pixels/frame)')

    # Plot 4: Feature categories bar chart
    categories = {
        'Temporal': len([k for k in features if any(x in k for x in ["stroke_rate", "cycle", "tempo"])]),
        'Kinematic': len([k for k in features if any(x in k for x in ["velocity", "acceleration", "path"])]),
        'Angular': len([k for k in features if any(x in k for x in ["elbow", "shoulder"]) and "asymmetry" not in k]),
        'Symmetry': len([k for k in features if "asymmetry" in k]),
        'Injury Risk': len([k for k in features if any(x in k for x in ["extreme", "drop", "risk", "workload"])]),
    }

    axes[1, 1].bar(categories.keys(), categories.values(), color=['blue', 'green', 'orange', 'purple', 'red'])
    axes[1, 1].set_title('Features by Category')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_dir = Path("data/exports")
    plt.savefig(output_dir / "phase3_features_demo.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Feature visualization saved to: {output_dir / 'phase3_features_demo.png'}")

    plt.close()

    return features


def demo_symmetry_analysis():
    """Demonstrate symmetry analysis."""
    print("\n" + "=" * 80)
    print("  DEMONSTRATION 3: Symmetry Analysis")
    print("=" * 80)

    # Generate synthetic data
    data = generate_synthetic_swimming_data(num_frames=150, fps=30.0)

    # Create analyzer
    analyzer = SymmetryAnalyzer(fps=data["fps"])

    # Perform comprehensive symmetry analysis
    print("\n⏳ Analyzing symmetry...")
    results = analyzer.comprehensive_symmetry_analysis(
        data["left_hand_path"],
        data["right_hand_path"],
        data["angles_over_time"],
    )

    print("✅ Symmetry analysis complete!")

    # Display results
    print("\n" + "=" * 80)
    print("  Overall Symmetry")
    print("=" * 80)
    print(f"\nSymmetry Score: {results['overall_symmetry_score']:.1f}/100")
    print(f"Interpretation: {results['interpretation']}")

    print("\n" + "=" * 80)
    print("  Arm Symmetry")
    print("=" * 80)
    arm_symmetry = results['arm_symmetry']
    for key, value in arm_symmetry.items():
        print(f"{key:<40} {value:>10.2f}")

    print("\n" + "=" * 80)
    print("  Temporal Symmetry")
    print("=" * 80)
    temporal_symmetry = results['temporal_symmetry']
    for key, value in temporal_symmetry.items():
        print(f"{key:<40} {value:>10.2f}")

    print("\n" + "=" * 80)
    print("  Force Imbalance")
    print("=" * 80)
    force_imbalance = results['force_imbalance']
    for key, value in force_imbalance.items():
        print(f"{key:<40} {value:>10.2f}")

    print("\n" + "=" * 80)
    print("  Recommendations")
    print("=" * 80)
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"\n{i}. {rec}")

    # Visualize symmetry results
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Hand trajectories (mirrored comparison)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data["left_hand_path"][:, 0], data["left_hand_path"][:, 1], 'b-', linewidth=2, label='Left Hand', alpha=0.7)
    ax1.plot(data["right_hand_path"][:, 0], data["right_hand_path"][:, 1], 'r-', linewidth=2, label='Right Hand', alpha=0.7)
    ax1.set_title('Hand Trajectories Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')

    # Plot 2: Path length comparison
    ax2 = fig.add_subplot(gs[1, 0])
    if "left_path_length" in arm_symmetry and "right_path_length" in arm_symmetry:
        lengths = [arm_symmetry["left_path_length"], arm_symmetry["right_path_length"]]
        bars = ax2.bar(['Left Arm', 'Right Arm'], lengths, color=['blue', 'red'], alpha=0.7)
        ax2.set_title('Path Length Comparison', fontweight='bold')
        ax2.set_ylabel('Path Length (pixels)')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)

    # Plot 3: Speed comparison
    ax3 = fig.add_subplot(gs[1, 1])
    if "left_mean_speed" in arm_symmetry and "right_mean_speed" in arm_symmetry:
        speeds = [arm_symmetry["left_mean_speed"], arm_symmetry["right_mean_speed"]]
        bars = ax3.bar(['Left Arm', 'Right Arm'], speeds, color=['blue', 'red'], alpha=0.7)
        ax3.set_title('Mean Speed Comparison', fontweight='bold')
        ax3.set_ylabel('Speed (pixels/frame)')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

    # Plot 4: Symmetry scores
    ax4 = fig.add_subplot(gs[2, 0])
    score_categories = []
    score_values = []

    if "path_length_asymmetry_pct" in arm_symmetry:
        score_categories.append('Path\nAsymmetry')
        score_values.append(arm_symmetry["path_length_asymmetry_pct"])

    if "speed_asymmetry_pct" in arm_symmetry:
        score_categories.append('Speed\nAsymmetry')
        score_values.append(arm_symmetry["speed_asymmetry_pct"])

    if "stroke_count_asymmetry_pct" in temporal_symmetry:
        score_categories.append('Stroke Count\nAsymmetry')
        score_values.append(temporal_symmetry["stroke_count_asymmetry_pct"])

    if score_values:
        colors = ['green' if v < 10 else 'orange' if v < 20 else 'red' for v in score_values]
        bars = ax4.bar(score_categories, score_values, color=colors, alpha=0.7)
        ax4.set_title('Asymmetry Metrics (%)', fontweight='bold')
        ax4.set_ylabel('Asymmetry (%)')
        ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Warning')
        ax4.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='High')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Overall score gauge
    ax5 = fig.add_subplot(gs[2, 1])
    overall_score = results['overall_symmetry_score']

    # Create gauge chart
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)

    # Color segments
    colors_map = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 100))
    for i in range(len(theta) - 1):
        ax5.fill_between(theta[i:i+2], 0, r[i:i+2], color=colors_map[i], alpha=0.7)

    # Add needle
    needle_angle = (1 - overall_score / 100) * np.pi
    ax5.plot([needle_angle, needle_angle], [0, 1], 'k-', linewidth=3)
    ax5.plot(needle_angle, 1, 'ko', markersize=10)

    ax5.set_ylim(0, 1.2)
    ax5.set_xlim(0, np.pi)
    ax5.axis('off')
    ax5.text(np.pi/2, 1.4, f'Overall Symmetry Score\n{overall_score:.1f}/100',
            ha='center', va='center', fontsize=14, fontweight='bold')

    plt.suptitle('Comprehensive Symmetry Analysis', fontsize=16, fontweight='bold')

    # Save figure
    output_dir = Path("data/exports")
    plt.savefig(output_dir / "phase3_symmetry_demo.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Symmetry visualization saved to: {output_dir / 'phase3_symmetry_demo.png'}")

    plt.close()


def main():
    """Run all Phase 3 demonstrations."""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "SWIMVISION PRO - PHASE 3")
    print(" " * 10 + "Biomechanical Feature Extraction & Symmetry Analysis")
    print("=" * 80)

    # Run demonstrations
    demo_smoothing()
    demo_feature_extraction()
    demo_symmetry_analysis()

    print("\n" + "=" * 80)
    print("  Phase 3 Demonstration Complete!")
    print("=" * 80)
    print("\n✅ All visualizations saved to: data/exports/")
    print("\n📊 Phase 3 Features:")
    print("   • Kalman & Savitzky-Golay filtering for trajectory smoothing")
    print("   • 30+ biomechanical features (temporal, kinematic, angular, symmetry)")
    print("   • Comprehensive symmetry analysis (arm, temporal, force balance)")
    print("   • Injury risk indicators")
    print("   • Automated recommendations")
    print("\n🎯 Next Steps:")
    print("   • Run the Streamlit app: streamlit run app.py")
    print("   • Try the 'Biomechanical Features' mode")
    print("   • Try the 'Symmetry Analysis' mode")
    print("   • Analyze your own swimming videos!")
    print("\n")


if __name__ == "__main__":
    main()
