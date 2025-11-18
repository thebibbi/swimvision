"""
SwimVision Pro - Phase 2 Demonstration Script

This script demonstrates all Phase 2 time-series analysis features with synthetic swimming data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.analysis.dtw_analyzer import DTWAnalyzer
from src.analysis.frechet_analyzer import FrechetAnalyzer
from src.analysis.similarity_measures import (
    SoftDTW,
    LCSS,
    CosineSimilarityAnalyzer,
    CrossCorrelationAnalyzer,
)
from src.analysis.stroke_phases import StrokePhaseDetector
from src.analysis.stroke_similarity import StrokeSimilarityEnsemble


def generate_synthetic_freestyle_stroke(
    num_frames: int = 90,
    stroke_rate: float = 1.0,
    variation: float = 0.1,
) -> dict:
    """Generate synthetic freestyle swimming stroke data.

    Args:
        num_frames: Number of frames to generate.
        stroke_rate: Strokes per second.
        variation: Amount of random variation (0-1).

    Returns:
        Dictionary with hand paths and angles.
    """
    # Generate time array
    fps = 30.0
    t = np.linspace(0, num_frames / fps, num_frames)

    # Generate hand path (elliptical pattern)
    # X-axis: forward/backward motion
    # Y-axis: up/down motion

    # Left hand path
    left_hand_x = 300 + 150 * np.sin(2 * np.pi * stroke_rate * t)
    left_hand_y = 250 + 100 * np.cos(2 * np.pi * stroke_rate * t)

    # Add variation
    left_hand_x += np.random.normal(0, variation * 10, num_frames)
    left_hand_y += np.random.normal(0, variation * 10, num_frames)

    left_hand_path = np.column_stack([left_hand_x, left_hand_y])

    # Right hand path (offset by half cycle)
    right_hand_x = 450 + 150 * np.sin(2 * np.pi * stroke_rate * t + np.pi)
    right_hand_y = 250 + 100 * np.cos(2 * np.pi * stroke_rate * t + np.pi)

    right_hand_x += np.random.normal(0, variation * 10, num_frames)
    right_hand_y += np.random.normal(0, variation * 10, num_frames)

    right_hand_path = np.column_stack([right_hand_x, right_hand_y])

    # Generate joint angles (sinusoidal patterns)
    left_elbow = 90 + 60 * np.sin(2 * np.pi * stroke_rate * t + np.pi/4)
    right_elbow = 90 + 60 * np.sin(2 * np.pi * stroke_rate * t + np.pi + np.pi/4)

    left_shoulder = 120 + 30 * np.sin(2 * np.pi * stroke_rate * t)
    right_shoulder = 120 + 30 * np.sin(2 * np.pi * stroke_rate * t + np.pi)

    # Add variation to angles
    left_elbow += np.random.normal(0, variation * 5, num_frames)
    right_elbow += np.random.normal(0, variation * 5, num_frames)
    left_shoulder += np.random.normal(0, variation * 3, num_frames)
    right_shoulder += np.random.normal(0, variation * 3, num_frames)

    return {
        "left_hand_path": left_hand_path.tolist(),
        "right_hand_path": right_hand_path.tolist(),
        "angles": {
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
        },
        "fps": fps,
    }


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demonstrate_dtw_analysis():
    """Demonstrate DTW analysis."""
    print_section_header("Dynamic Time Warping (DTW) Analysis")

    analyzer = DTWAnalyzer()

    # Generate two similar strokes
    stroke1 = generate_synthetic_freestyle_stroke(variation=0.05)
    stroke2 = generate_synthetic_freestyle_stroke(variation=0.10)

    left_path1 = np.array(stroke1["left_hand_path"])
    left_path2 = np.array(stroke2["left_hand_path"])

    # Compute DTW distance
    distance = analyzer.compute_distance(left_path1, left_path2, normalize=True)
    similarity_score = analyzer.compute_similarity_score(distance, max_distance=10.0)

    print(f"DTW Distance: {distance:.4f}")
    print(f"Similarity Score: {similarity_score:.2f}/100")
    print(f"Interpretation: {'Excellent match!' if similarity_score > 90 else 'Good match' if similarity_score > 75 else 'Moderate similarity'}")

    # Get warping path
    path = analyzer.get_warping_path(left_path1, left_path2)
    print(f"\nWarping path length: {len(path)}")
    print(f"Alignment efficiency: {len(path) / max(len(left_path1), len(left_path2)):.2f}")


def demonstrate_similarity_measures():
    """Demonstrate additional similarity measures."""
    print_section_header("Additional Similarity Measures")

    # Generate test sequences
    stroke1 = generate_synthetic_freestyle_stroke(variation=0.05)
    stroke2 = generate_synthetic_freestyle_stroke(variation=0.08)

    left_path1 = np.array(stroke1["left_hand_path"])
    left_path2 = np.array(stroke2["left_hand_path"])

    # Soft-DTW
    print("1. Soft-DTW (Differentiable Variant)")
    soft_dtw = SoftDTW(gamma=1.0)
    soft_distance = soft_dtw.compute_distance(left_path1, left_path2)
    print(f"   Soft-DTW Distance: {soft_distance:.4f}")

    # LCSS
    print("\n2. LCSS (Longest Common Subsequence)")
    lcss = LCSS(epsilon=10.0, delta=5)
    lcss_sim = lcss.compute_similarity(left_path1, left_path2, normalize=True)
    print(f"   LCSS Similarity: {lcss_sim:.4f} (0-1 scale)")
    print(f"   LCSS Score: {lcss_sim * 100:.2f}/100")

    # Cosine Similarity (on angles)
    print("\n3. Cosine Similarity (Joint Angles)")
    angles1 = stroke1["angles"]["left_elbow"]
    angles2 = stroke2["angles"]["left_elbow"]
    cosine_sim = CosineSimilarityAnalyzer.compute_similarity(angles1, angles2)
    print(f"   Cosine Similarity: {cosine_sim:.4f}")
    print(f"   Angle alignment: {'Excellent' if cosine_sim > 0.9 else 'Good' if cosine_sim > 0.7 else 'Moderate'}")

    # Cross-Correlation
    print("\n4. Cross-Correlation (Phase Alignment)")
    shift, max_corr = CrossCorrelationAnalyzer.find_best_alignment(
        left_path1[:, 0], left_path2[:, 0]
    )
    print(f"   Optimal time shift: {shift} frames")
    print(f"   Max correlation: {max_corr:.4f}")
    print(f"   Temporal alignment: {'Synchronized' if abs(shift) < 5 else 'Slight offset'}")


def demonstrate_frechet_analysis():
    """Demonstrate Fréchet distance analysis."""
    print_section_header("Fréchet Distance Analysis")

    analyzer = FrechetAnalyzer()

    # Generate strokes
    stroke1 = generate_synthetic_freestyle_stroke(variation=0.05)
    stroke2 = generate_synthetic_freestyle_stroke(variation=0.12)

    left_path1 = np.array(stroke1["left_hand_path"])
    left_path2 = np.array(stroke2["left_hand_path"])

    # Compute Fréchet distance
    frechet_distance = analyzer.compute_distance(left_path1, left_path2)
    print(f"Fréchet Distance: {frechet_distance:.2f} pixels")

    # Analyze trajectory shape
    print("\nTrajectory Shape Analysis:")
    shape1 = analyzer.analyze_trajectory_shape(left_path1)
    shape2 = analyzer.analyze_trajectory_shape(left_path2)

    print(f"\n  Stroke 1:")
    print(f"    Path length: {shape1['path_length']:.2f} pixels")
    print(f"    Path efficiency: {shape1['path_efficiency']:.3f}")
    print(f"    Bounding box: {shape1['bbox_width']:.1f} x {shape1['bbox_height']:.1f}")

    print(f"\n  Stroke 2:")
    print(f"    Path length: {shape2['path_length']:.2f} pixels")
    print(f"    Path efficiency: {shape2['path_efficiency']:.3f}")
    print(f"    Bounding box: {shape2['bbox_width']:.1f} x {shape2['bbox_height']:.1f}")

    # Classify pull patterns
    print("\nPull Pattern Classification:")
    pattern1 = analyzer.classify_pull_pattern(left_path1)
    pattern2 = analyzer.classify_pull_pattern(left_path2)

    print(f"  Stroke 1: {pattern1}")
    print(f"  Stroke 2: {pattern2}")
    print(f"  Patterns match: {pattern1 == pattern2}")


def demonstrate_stroke_phases():
    """Demonstrate stroke phase detection."""
    print_section_header("Stroke Phase Detection")

    detector = StrokePhaseDetector()

    # Generate stroke
    stroke = generate_synthetic_freestyle_stroke(num_frames=120, variation=0.08)
    left_path = np.array(stroke["left_hand_path"])
    fps = stroke["fps"]

    # Detect phases
    phases = detector.detect_phases_freestyle(left_path, fps, "left")

    print(f"Detected {len(phases)} stroke phases:")
    print(f"\n{'Phase':<12} {'Start':<8} {'End':<8} {'Duration (s)':<12}")
    print("-" * 50)

    for phase in phases:
        print(f"{phase['phase'].value:<12} {phase['start_frame']:<8} {phase['end_frame']:<8} {phase['duration']:.3f}")

    # Phase durations
    print("\nAverage Phase Durations:")
    phase_durations = detector.get_phase_durations(phases)

    for phase_name, duration in phase_durations.items():
        print(f"  {phase_name}: {duration:.3f}s")

    # Detect cycle boundaries
    cycles = detector.detect_cycle_boundaries(left_path, fps)
    print(f"\nDetected {len(cycles)} complete stroke cycles")


def demonstrate_comprehensive_comparison():
    """Demonstrate comprehensive stroke comparison ensemble."""
    print_section_header("Comprehensive Stroke Comparison (Ensemble)")

    ensemble = StrokeSimilarityEnsemble()

    # Generate reference and test strokes
    print("Generating strokes...")
    print("  - Reference stroke (ideal technique)")
    print("  - Test stroke (swimmer's actual stroke)")

    reference_stroke = generate_synthetic_freestyle_stroke(variation=0.03)
    test_stroke = generate_synthetic_freestyle_stroke(variation=0.15)

    # Perform comprehensive comparison
    print("\nAnalyzing similarity using ensemble of 6 algorithms...")
    results = ensemble.comprehensive_comparison(
        reference_stroke,
        test_stroke,
        fps=30.0,
    )

    # Display overall score
    print(f"\n{'OVERALL SIMILARITY SCORE':<30} {results['overall_score']:.1f}/100")

    # Display individual scores
    print("\nIndividual Metric Scores:")
    print(f"{'Metric':<30} {'Score':>10}")
    print("-" * 42)

    for metric, score in results["individual_scores"].items():
        print(f"{metric.replace('_', ' ').title():<30} {score:>10.1f}")

    # Display additional measures
    print("\nAdditional Similarity Measures:")
    for measure, value in results["additional_measures"].items():
        print(f"  {measure.replace('_', ' ').title()}: {value:.4f}")

    # Display recommendations
    print("\n" + "=" * 80)
    print("TECHNIQUE RECOMMENDATIONS")
    print("=" * 80)

    for i, rec in enumerate(results["recommendations"], 1):
        print(f"\n{i}. {rec}")


def demonstrate_progressive_analysis():
    """Demonstrate progressive analysis for fatigue detection."""
    print_section_header("Progressive Analysis (Fatigue Detection)")

    ensemble = StrokeSimilarityEnsemble()

    # Generate sequence of strokes with increasing variation (simulating fatigue)
    print("Simulating 10 consecutive strokes with progressive fatigue...\n")

    stroke_sequence = []
    for i in range(10):
        variation = 0.05 + (i * 0.02)  # Increasing variation
        stroke = generate_synthetic_freestyle_stroke(variation=variation)
        stroke_sequence.append(stroke)

    # Analyze progression
    results = ensemble.progressive_analysis(stroke_sequence, fps=30.0)

    print(f"Total strokes analyzed: {results['stroke_count']}")
    print(f"\nConsistency scores between consecutive strokes:")

    for i, score in enumerate(results["consistency_scores"], 1):
        bar = "█" * int(score / 5)
        print(f"  Stroke {i} → {i+1}: {score:>5.1f}/100 {bar}")

    print(f"\nTrend analysis: {results['trend'].upper()}")

    if results['fatigue_detected']:
        print("⚠️  FATIGUE DETECTED: Technique degradation observed")
        print("   Recommendation: Take a break to maintain form")
    else:
        print("✅ No significant fatigue detected")

    if 'trend_slope' in results:
        print(f"   Trend slope: {results['trend_slope']:.2f} points/stroke")


def create_visualization():
    """Create visual comparison of two strokes."""
    print_section_header("Creating Visualization")

    # Generate two strokes
    stroke1 = generate_synthetic_freestyle_stroke(variation=0.05)
    stroke2 = generate_synthetic_freestyle_stroke(variation=0.12)

    left_path1 = np.array(stroke1["left_hand_path"])
    left_path2 = np.array(stroke2["left_hand_path"])

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SwimVision Pro - Phase 2 Demonstration", fontsize=16, fontweight='bold')

    # Hand trajectories
    axes[0, 0].plot(left_path1[:, 0], -left_path1[:, 1], 'b-o', label='Reference Stroke', markersize=3)
    axes[0, 0].plot(left_path2[:, 0], -left_path2[:, 1], 'r-o', label='Test Stroke', markersize=3, alpha=0.7)
    axes[0, 0].set_title("Hand Trajectories Comparison")
    axes[0, 0].set_xlabel("X Position (pixels)")
    axes[0, 0].set_ylabel("Y Position (pixels)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Joint angles over time
    angles1 = stroke1["angles"]["left_elbow"]
    angles2 = stroke2["angles"]["left_elbow"]

    axes[0, 1].plot(angles1, 'b-', label='Reference Stroke', linewidth=2)
    axes[0, 1].plot(angles2, 'r-', label='Test Stroke', linewidth=2, alpha=0.7)
    axes[0, 1].set_title("Left Elbow Angle Over Time")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Angle (degrees)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Similarity scores comparison
    analyzer = DTWAnalyzer()
    soft_dtw = SoftDTW(gamma=1.0)
    lcss = LCSS(epsilon=10.0, delta=5)

    dtw_dist = analyzer.compute_distance(left_path1, left_path2, normalize=True)
    dtw_score = analyzer.compute_similarity_score(dtw_dist, max_distance=10.0)
    soft_score = (1.0 - min(soft_dtw.compute_distance(left_path1, left_path2) / 50.0, 1.0)) * 100
    lcss_score = lcss.compute_similarity(left_path1, left_path2, normalize=True) * 100

    scores = [dtw_score, soft_score, lcss_score]
    methods = ['DTW', 'Soft-DTW', 'LCSS']

    axes[1, 0].bar(methods, scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1, 0].set_title("Similarity Scores by Method")
    axes[1, 0].set_ylabel("Similarity Score (0-100)")
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1, 0].axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Text summary
    axes[1, 1].axis('off')
    summary_text = f"""
    ANALYSIS SUMMARY
    ═══════════════════════════════════════

    Overall Similarity: {np.mean(scores):.1f}/100

    Individual Scores:
      • DTW Score: {dtw_score:.1f}/100
      • Soft-DTW Score: {soft_score:.1f}/100
      • LCSS Score: {lcss_score:.1f}/100

    Assessment:
      {"✅ Excellent technique match!" if np.mean(scores) > 90 else "👍 Good similarity" if np.mean(scores) > 75 else "⚠️  Moderate differences detected"}

    Recommendations:
      • Focus on trajectory consistency
      • Maintain elbow angle range
      • Work on stroke rhythm

    Phase 2 Features Demonstrated:
      ✓ DTW Analysis
      ✓ Multiple Similarity Measures
      ✓ Trajectory Comparison
      ✓ Angle Analysis
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_path = Path("data/exports/phase2_demonstration.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")

    plt.close()


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "SwimVision Pro - Phase 2 Demo" + " " * 29 + "║")
    print("║" + " " * 15 + "Time-Series Analysis & Stroke Comparison" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")

    # Run all demonstrations
    demonstrate_dtw_analysis()
    demonstrate_similarity_measures()
    demonstrate_frechet_analysis()
    demonstrate_stroke_phases()
    demonstrate_comprehensive_comparison()
    demonstrate_progressive_analysis()

    # Create visualization
    create_visualization()

    # Final summary
    print_section_header("Demonstration Complete!")
    print("✅ All Phase 2 features demonstrated successfully!")
    print("\nFeatures showcased:")
    print("  1. Dynamic Time Warping (DTW)")
    print("  2. Soft-DTW (differentiable variant)")
    print("  3. LCSS (Longest Common Subsequence)")
    print("  4. Fréchet Distance for trajectories")
    print("  5. Cosine Similarity for angles")
    print("  6. Cross-Correlation for phase alignment")
    print("  7. Stroke Phase Detection (5 phases)")
    print("  8. Pull Pattern Classification")
    print("  9. Comprehensive Ensemble Scoring")
    print(" 10. Progressive Analysis (fatigue detection)")
    print("\nCheck data/exports/phase2_demonstration.png for visualizations!")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
