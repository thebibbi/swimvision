"""Phase template system for ideal underwater stroke paths.

Stores and retrieves ideal trajectories for each stroke phase,
used to guide predictions during occlusion.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pickle
import json

from src.analysis.stroke_phases import StrokePhase


@dataclass
class PhaseTemplate:
    """Template for a specific stroke phase."""
    phase: StrokePhase
    stroke_type: str  # 'freestyle', 'backstroke', 'breaststroke', 'butterfly'
    hand_side: str    # 'left' or 'right'
    trajectory: np.ndarray  # Nx2 array of (x, y) positions
    duration_frames: int    # Typical duration in frames
    velocity_profile: Optional[np.ndarray] = None  # Nx2 velocity vectors
    metadata: Optional[Dict] = None


class PhaseTemplateManager:
    """Manage stroke phase templates."""

    def __init__(self, template_dir: str = "models/phase_templates"):
        """Initialize template manager.

        Args:
            template_dir: Directory to store templates.
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Template storage
        self.templates: Dict[str, PhaseTemplate] = {}

        # Load existing templates
        self.load_all_templates()

    def create_template(
        self,
        phase: StrokePhase,
        stroke_type: str,
        hand_side: str,
        trajectory: np.ndarray,
        fps: float = 30.0,
        metadata: Optional[Dict] = None,
    ) -> PhaseTemplate:
        """Create a new phase template.

        Args:
            phase: Stroke phase.
            stroke_type: Type of stroke.
            hand_side: Left or right hand.
            trajectory: Trajectory points (Nx2).
            fps: Video frame rate.
            metadata: Additional metadata.

        Returns:
            Created template.
        """
        # Calculate duration
        duration_frames = len(trajectory)

        # Calculate velocity profile
        velocity_profile = None
        if len(trajectory) > 1:
            velocities = np.diff(trajectory, axis=0) * fps
            # Pad to match trajectory length
            velocity_profile = np.vstack([velocities, velocities[-1:]])

        # Create template
        template = PhaseTemplate(
            phase=phase,
            stroke_type=stroke_type,
            hand_side=hand_side,
            trajectory=trajectory,
            duration_frames=duration_frames,
            velocity_profile=velocity_profile,
            metadata=metadata or {},
        )

        # Store template
        key = self._get_template_key(stroke_type, phase, hand_side)
        self.templates[key] = template

        return template

    def get_template(
        self,
        stroke_type: str,
        phase: StrokePhase,
        hand_side: str,
    ) -> Optional[PhaseTemplate]:
        """Get template for specific stroke, phase, and hand.

        Args:
            stroke_type: Type of stroke.
            phase: Stroke phase.
            hand_side: Left or right hand.

        Returns:
            Template if found, None otherwise.
        """
        key = self._get_template_key(stroke_type, phase, hand_side)
        return self.templates.get(key)

    def interpolate_template(
        self,
        template: PhaseTemplate,
        progress: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Interpolate position along template based on progress.

        Args:
            template: Phase template.
            progress: Progress through phase (0.0 to 1.0).

        Returns:
            Tuple of (position, velocity).
        """
        progress = np.clip(progress, 0.0, 1.0)

        # Calculate index in trajectory
        idx = int(progress * (len(template.trajectory) - 1))
        idx = min(idx, len(template.trajectory) - 1)

        position = template.trajectory[idx]

        velocity = None
        if template.velocity_profile is not None:
            velocity = template.velocity_profile[idx]

        return position, velocity

    def save_template(self, template: PhaseTemplate, filename: Optional[str] = None):
        """Save template to file.

        Args:
            template: Template to save.
            filename: Optional filename (auto-generated if None).
        """
        if filename is None:
            filename = f"{template.stroke_type}_{template.phase.value}_{template.hand_side}.pkl"

        filepath = self.template_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump(template, f)

    def load_template(self, filename: str) -> Optional[PhaseTemplate]:
        """Load template from file.

        Args:
            filename: Filename to load.

        Returns:
            Loaded template or None if failed.
        """
        filepath = self.template_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'rb') as f:
                template = pickle.load(f)
            return template
        except Exception as e:
            print(f"Failed to load template {filename}: {e}")
            return None

    def load_all_templates(self):
        """Load all templates from template directory."""
        if not self.template_dir.exists():
            return

        for filepath in self.template_dir.glob("*.pkl"):
            template = self.load_template(filepath.name)
            if template:
                key = self._get_template_key(
                    template.stroke_type,
                    template.phase,
                    template.hand_side,
                )
                self.templates[key] = template

    def save_all_templates(self):
        """Save all templates to files."""
        for template in self.templates.values():
            self.save_template(template)

    def create_from_recorded_stroke(
        self,
        full_trajectory: np.ndarray,
        phase_segments: List[Dict],
        stroke_type: str,
        hand_side: str,
        fps: float = 30.0,
    ):
        """Create templates from a recorded stroke cycle.

        Args:
            full_trajectory: Full stroke trajectory (Nx2).
            phase_segments: List of phase segments with start/end frames.
            stroke_type: Type of stroke.
            hand_side: Left or right hand.
            fps: Frame rate.

        Example phase_segments:
        [
            {'phase': StrokePhase.ENTRY, 'start_frame': 0, 'end_frame': 10},
            {'phase': StrokePhase.CATCH, 'start_frame': 10, 'end_frame': 20},
            ...
        ]
        """
        for segment in phase_segments:
            phase = segment['phase']
            start = segment['start_frame']
            end = segment['end_frame']

            # Extract phase trajectory
            phase_trajectory = full_trajectory[start:end]

            if len(phase_trajectory) < 2:
                continue

            # Create template
            self.create_template(
                phase=phase,
                stroke_type=stroke_type,
                hand_side=hand_side,
                trajectory=phase_trajectory,
                fps=fps,
                metadata={
                    'source': 'recorded',
                    'start_frame': start,
                    'end_frame': end,
                },
            )

    def generate_ideal_freestyle_templates(self, fps: float = 30.0):
        """Generate ideal templates for freestyle stroke.

        Uses biomechanical knowledge to create reference templates.

        Args:
            fps: Frame rate.
        """
        # Entry phase (hand entering water)
        entry_traj = self._generate_entry_trajectory(fps)
        self.create_template(
            StrokePhase.ENTRY,
            'freestyle',
            'left',
            entry_traj,
            fps,
            {'source': 'generated'},
        )

        # Catch phase (underwater, setting up pull)
        catch_traj = self._generate_catch_trajectory(fps)
        self.create_template(
            StrokePhase.CATCH,
            'freestyle',
            'left',
            catch_traj,
            fps,
            {'source': 'generated'},
        )

        # Pull phase (main propulsion)
        pull_traj = self._generate_pull_trajectory(fps)
        self.create_template(
            StrokePhase.PULL,
            'freestyle',
            'left',
            pull_traj,
            fps,
            {'source': 'generated'},
        )

        # Push phase (completing stroke)
        push_traj = self._generate_push_trajectory(fps)
        self.create_template(
            StrokePhase.PUSH,
            'freestyle',
            'left',
            push_traj,
            fps,
            {'source': 'generated'},
        )

        # Recovery phase (arm out of water)
        recovery_traj = self._generate_recovery_trajectory(fps)
        self.create_template(
            StrokePhase.RECOVERY,
            'freestyle',
            'left',
            recovery_traj,
            fps,
            {'source': 'generated'},
        )

        # Mirror for right hand
        for phase in StrokePhase:
            left_template = self.get_template('freestyle', phase, 'left')
            if left_template:
                # Mirror x-coordinates
                right_traj = left_template.trajectory.copy()
                right_traj[:, 0] = -right_traj[:, 0]

                self.create_template(
                    phase,
                    'freestyle',
                    'right',
                    right_traj,
                    fps,
                    {'source': 'generated_mirror'},
                )

    def _generate_entry_trajectory(self, fps: float) -> np.ndarray:
        """Generate entry phase trajectory."""
        # Entry: hand moves forward and down
        duration = int(0.3 * fps)  # ~0.3 seconds
        t = np.linspace(0, 1, duration)

        x = 150 + 30 * t  # Forward
        y = 100 + 80 * t  # Downward

        return np.column_stack([x, y])

    def _generate_catch_trajectory(self, fps: float) -> np.ndarray:
        """Generate catch phase trajectory."""
        # Catch: hand slows, sets up for pull
        duration = int(0.2 * fps)
        t = np.linspace(0, 1, duration)

        x = 180 + 10 * (1 - t)  # Slight forward
        y = 180 + 30 * t        # Deeper

        return np.column_stack([x, y])

    def _generate_pull_trajectory(self, fps: float) -> np.ndarray:
        """Generate pull phase trajectory."""
        # Pull: main propulsive phase, backward and down
        duration = int(0.4 * fps)
        t = np.linspace(0, 1, duration)

        x = 190 - 90 * t  # Backward (main propulsion)
        y = 210 + 40 * np.sin(np.pi * t)  # S-curve depth

        return np.column_stack([x, y])

    def _generate_push_trajectory(self, fps: float) -> np.ndarray:
        """Generate push phase trajectory."""
        # Push: finishing stroke, backward and up
        duration = int(0.3 * fps)
        t = np.linspace(0, 1, duration)

        x = 100 - 40 * t  # Continuing backward
        y = 250 - 30 * t  # Rising

        return np.column_stack([x, y])

    def _generate_recovery_trajectory(self, fps: float) -> np.ndarray:
        """Generate recovery phase trajectory."""
        # Recovery: hand out of water, forward
        duration = int(0.5 * fps)
        t = np.linspace(0, 1, duration)

        x = 60 + 90 * t   # Forward (recovering)
        y = 220 - 120 * t  # Upward and forward

        return np.column_stack([x, y])

    def export_templates_json(self, filename: str):
        """Export templates to JSON for visualization/analysis.

        Args:
            filename: Output filename.
        """
        export_data = {}

        for key, template in self.templates.items():
            export_data[key] = {
                'stroke_type': template.stroke_type,
                'phase': template.phase.value,
                'hand_side': template.hand_side,
                'trajectory': template.trajectory.tolist(),
                'duration_frames': template.duration_frames,
                'metadata': template.metadata,
            }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

    def _get_template_key(
        self,
        stroke_type: str,
        phase: StrokePhase,
        hand_side: str,
    ) -> str:
        """Generate unique key for template.

        Args:
            stroke_type: Type of stroke.
            phase: Stroke phase.
            hand_side: Left or right hand.

        Returns:
            Template key string.
        """
        return f"{stroke_type}_{phase.value}_{hand_side}"

    def get_statistics(self) -> Dict:
        """Get template statistics.

        Returns:
            Dictionary with statistics.
        """
        stats = {
            'total_templates': len(self.templates),
            'by_stroke': {},
            'by_phase': {},
            'by_hand': {},
        }

        for template in self.templates.values():
            # By stroke
            if template.stroke_type not in stats['by_stroke']:
                stats['by_stroke'][template.stroke_type] = 0
            stats['by_stroke'][template.stroke_type] += 1

            # By phase
            phase_name = template.phase.value
            if phase_name not in stats['by_phase']:
                stats['by_phase'][phase_name] = 0
            stats['by_phase'][phase_name] += 1

            # By hand
            if template.hand_side not in stats['by_hand']:
                stats['by_hand'][template.hand_side] = 0
            stats['by_hand'][template.hand_side] += 1

        return stats
