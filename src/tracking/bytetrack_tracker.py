"""ByteTrack wrapper for multi-swimmer tracking.

ByteTrack is a simple yet effective multi-object tracking method that:
- Uses ALL detections (high and low confidence)
- Associates tracks using IoU and Kalman filtering
- Handles occlusions and re-identification well

Perfect for tracking swimmers who frequently go underwater.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    from filterpy.kalman import KalmanFilter

    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False


class TrackState(Enum):
    """Track state enumeration."""

    NEW = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


@dataclass
class SwimmerTrack:
    """Tracked swimmer data."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    keypoints: np.ndarray | None = None  # (17, 3)
    confidence: float = 0.0
    state: TrackState = TrackState.NEW
    frame_id: int = 0
    age: int = 0  # Total frames this track has existed
    hits: int = 0  # Consecutive frames with detection
    time_since_update: int = 0  # Frames since last update

    # Trajectory history
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    keypoint_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Kalman filter for bbox prediction
    kalman_filter: object | None = None

    def __post_init__(self):
        """Initialize Kalman filter after creation."""
        if FILTERPY_AVAILABLE and self.kalman_filter is None:
            self.kalman_filter = self._create_kalman_filter()

    def _create_kalman_filter(self) -> KalmanFilter:
        """Create Kalman filter for bbox tracking.

        State: [x_center, y_center, area, aspect_ratio, vx, vy, va, vr]
        Measurement: [x_center, y_center, area, aspect_ratio]
        """
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
                [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
                [0, 0, 1, 0, 0, 0, 1, 0],  # a = a + va
                [0, 0, 0, 1, 0, 0, 0, 1],  # r = r + vr
                [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
                [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
                [0, 0, 0, 0, 0, 0, 1, 0],  # va = va
                [0, 0, 0, 0, 0, 0, 0, 1],  # vr = vr
            ]
        )

        # Measurement function (observe position and size)
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        # Measurement noise
        kf.R *= 10.0

        # Process noise
        kf.Q[-4:, -4:] *= 0.01  # Low velocity noise
        kf.Q[:4, :4] *= 0.1  # Position/size noise

        # Initial state covariance
        kf.P *= 10.0
        kf.P[-4:, -4:] *= 1000.0  # High velocity uncertainty initially

        return kf

    def predict(self) -> np.ndarray:
        """Predict next bbox using Kalman filter.

        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        if self.kalman_filter is None:
            return self.bbox

        # Predict
        self.kalman_filter.predict()

        # Convert state to bbox
        state = self.kalman_filter.x
        x_center, y_center, area, aspect_ratio = state[:4]

        w = np.sqrt(area * aspect_ratio)
        h = area / w

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        return np.array([x1, y1, x2, y2])

    def update(
        self, bbox: np.ndarray, keypoints: np.ndarray | None = None, confidence: float = 0.0
    ):
        """Update track with new detection.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            keypoints: Keypoints (17, 3)
            confidence: Detection confidence
        """
        self.bbox = bbox
        self.keypoints = keypoints
        self.confidence = confidence
        self.time_since_update = 0
        self.hits += 1

        # Update history
        self.bbox_history.append(bbox.copy())
        if keypoints is not None:
            self.keypoint_history.append(keypoints.copy())

        # Update Kalman filter
        if self.kalman_filter is not None:
            # Convert bbox to measurement
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2
            area = w * h
            aspect_ratio = w / (h + 1e-6)

            measurement = np.array([x_center, y_center, area, aspect_ratio])

            # Initialize filter on first update
            if self.age == 0:
                self.kalman_filter.x[:4] = measurement
            else:
                self.kalman_filter.update(measurement)

        # Update state
        if self.state == TrackState.NEW and self.hits >= 3 or self.state == TrackState.LOST:
            self.state = TrackState.TRACKED

    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1

        if self.state == TrackState.NEW or self.time_since_update > 30:
            self.state = TrackState.REMOVED
        elif self.time_since_update > 5:  # Lost for 5 frames
            self.state = TrackState.LOST

    def get_velocity(self) -> tuple[float, float]:
        """Get current velocity from Kalman filter.

        Returns:
            Tuple of (vx, vy)
        """
        if self.kalman_filter is None:
            return (0.0, 0.0)

        return (float(self.kalman_filter.x[4]), float(self.kalman_filter.x[5]))


class ByteTrackTracker:
    """ByteTrack multi-swimmer tracker.

    Features:
    - High/low confidence detection association
    - Kalman filter prediction
    - IoU-based matching
    - Track lifecycle management
    """

    def __init__(
        self,
        track_thresh: float = 0.5,  # High confidence threshold
        track_buffer: int = 30,  # Frames to keep lost tracks
        match_thresh: float = 0.8,  # IoU threshold for matching
        min_box_area: float = 100.0,  # Minimum bbox area
    ):
        """Initialize ByteTrack tracker.

        Args:
            track_thresh: Confidence threshold for track initiation
            track_buffer: Number of frames to buffer lost tracks
            match_thresh: IoU threshold for data association
            min_box_area: Minimum bounding box area
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        # Track management
        self.tracked_tracks: list[SwimmerTrack] = []
        self.lost_tracks: list[SwimmerTrack] = []
        self.removed_tracks: list[SwimmerTrack] = []

        self.frame_id = 0
        self.track_id_count = 0

    def update(
        self,
        detections: list[dict],
        frame_id: int | None = None,
    ) -> list[SwimmerTrack]:
        """Update tracks with new detections.

        Args:
            detections: List of detection dictionaries with 'bbox', 'keypoints', 'confidence'
            frame_id: Current frame ID (auto-incremented if None)

        Returns:
            List of active tracks
        """
        if frame_id is not None:
            self.frame_id = frame_id
        else:
            self.frame_id += 1

        # Separate high and low confidence detections
        high_dets = []
        low_dets = []

        for det in detections:
            bbox = np.array(det.get("bbox", [0, 0, 0, 0]))
            keypoints = det.get("keypoints")
            conf = det.get("confidence", np.mean(keypoints[:, 2]) if keypoints is not None else 0.0)

            # Filter by area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_box_area:
                continue

            det_data = {
                "bbox": bbox,
                "keypoints": keypoints,
                "confidence": conf,
            }

            if conf >= self.track_thresh:
                high_dets.append(det_data)
            else:
                low_dets.append(det_data)

        # Step 1: Match high confidence detections to tracked tracks
        tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        unmatched_tracks, matched_tracks, unmatched_high_dets = self._match_detections(
            tracked_tracks, high_dets, self.match_thresh
        )

        # Step 2: Match low confidence detections to unmatched tracks
        unmatched_tracks, matched_low_tracks, unmatched_low_dets = self._match_detections(
            unmatched_tracks,
            low_dets,
            0.5,  # Lower threshold for low conf
        )

        # Update matched tracks
        for track, det in matched_tracks + matched_low_tracks:
            track.update(det["bbox"], det["keypoints"], det["confidence"])
            track.frame_id = self.frame_id
            track.age += 1

        # Step 3: Try to match unmatched high confidence detections to lost tracks
        lost_tracks = [t for t in self.lost_tracks if t.state == TrackState.LOST]
        unmatched_lost, matched_lost, remaining_high_dets = self._match_detections(
            lost_tracks,
            unmatched_high_dets,
            0.6,  # More lenient for lost tracks
        )

        # Reactivate matched lost tracks
        for track, det in matched_lost:
            track.update(det["bbox"], det["keypoints"], det["confidence"])
            track.frame_id = self.frame_id
            track.age += 1
            self.tracked_tracks.append(track)
            self.lost_tracks.remove(track)

        # Step 4: Create new tracks for remaining high confidence detections
        for det in remaining_high_dets:
            new_track = SwimmerTrack(
                track_id=self._get_new_id(),
                bbox=det["bbox"],
                keypoints=det["keypoints"],
                confidence=det["confidence"],
                state=TrackState.NEW,
                frame_id=self.frame_id,
                age=1,
                hits=1,
            )
            self.tracked_tracks.append(new_track)

        # Mark unmatched tracks as missed
        for track in unmatched_tracks + unmatched_lost:
            track.mark_missed()

            # Move to lost/removed
            if track.state == TrackState.REMOVED:
                if track in self.tracked_tracks:
                    self.tracked_tracks.remove(track)
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                self.removed_tracks.append(track)
            elif track.state == TrackState.LOST:
                if track in self.tracked_tracks:
                    self.tracked_tracks.remove(track)
                if track not in self.lost_tracks:
                    self.lost_tracks.append(track)

        # Clean up old removed tracks
        self.removed_tracks = [
            t for t in self.removed_tracks if self.frame_id - t.frame_id < self.track_buffer
        ]

        # Return active tracks
        return [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]

    def _match_detections(
        self,
        tracks: list[SwimmerTrack],
        detections: list[dict],
        iou_threshold: float,
    ) -> tuple[list[SwimmerTrack], list[tuple[SwimmerTrack, dict]], list[dict]]:
        """Match detections to tracks using IoU.

        Args:
            tracks: List of tracks
            detections: List of detections
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (unmatched_tracks, matched_pairs, unmatched_detections)
        """
        if len(tracks) == 0:
            return [], [], detections

        if len(detections) == 0:
            return tracks, [], []

        # Get predicted bboxes for tracks
        track_bboxes = np.array([t.predict() for t in tracks])
        det_bboxes = np.array([d["bbox"] for d in detections])

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)

        # Hungarian matching (greedy approximation)
        matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
            iou_matrix, iou_threshold
        )

        # Build matched pairs
        matched_pairs = [(tracks[track_idx], detections[det_idx]) for track_idx, det_idx in matches]

        # Build unmatched lists
        unmatched_tracks_list = [tracks[idx] for idx in unmatched_tracks]
        unmatched_dets_list = [detections[idx] for idx in unmatched_dets]

        return unmatched_tracks_list, matched_pairs, unmatched_dets_list

    def _compute_iou_matrix(
        self,
        bboxes1: np.ndarray,
        bboxes2: np.ndarray,
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of bboxes.

        Args:
            bboxes1: (N, 4) array of bboxes
            bboxes2: (M, 4) array of bboxes

        Returns:
            (N, M) IoU matrix
        """
        N = len(bboxes1)
        M = len(bboxes2)

        iou_matrix = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                iou_matrix[i, j] = self._compute_iou(bboxes1[i], bboxes2[j])

        return iou_matrix

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy linear assignment (simplified Hungarian).

        Args:
            cost_matrix: (N, M) cost matrix (higher is better for IoU)
            threshold: Minimum cost threshold

        Returns:
            Tuple of (matches, unmatched_rows, unmatched_cols)
        """
        matches = []
        unmatched_rows = list(range(cost_matrix.shape[0]))
        unmatched_cols = list(range(cost_matrix.shape[1]))

        # Greedy matching: iteratively take highest cost pairs
        while len(unmatched_rows) > 0 and len(unmatched_cols) > 0:
            # Find maximum cost
            max_cost = -1
            max_row, max_col = -1, -1

            for i in unmatched_rows:
                for j in unmatched_cols:
                    if cost_matrix[i, j] > max_cost:
                        max_cost = cost_matrix[i, j]
                        max_row, max_col = i, j

            # If best cost is below threshold, stop
            if max_cost < threshold:
                break

            # Add match
            matches.append((max_row, max_col))
            unmatched_rows.remove(max_row)
            unmatched_cols.remove(max_col)

        return matches, unmatched_rows, unmatched_cols

    def _get_new_id(self) -> int:
        """Get new track ID."""
        self.track_id_count += 1
        return self.track_id_count

    def get_track_history(self, track_id: int, history_len: int = 30) -> dict | None:
        """Get trajectory history for a track.

        Args:
            track_id: Track ID
            history_len: Number of frames of history

        Returns:
            Dictionary with bbox and keypoint history
        """
        # Find track
        track = None
        for t in self.tracked_tracks + self.lost_tracks:
            if t.track_id == track_id:
                track = t
                break

        if track is None:
            return None

        return {
            "track_id": track_id,
            "bbox_history": list(track.bbox_history)[-history_len:],
            "keypoint_history": list(track.keypoint_history)[-history_len:],
            "age": track.age,
            "state": track.state.name,
        }

    def reset(self):
        """Reset tracker."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0
        self.track_id_count = 0


def demo_bytetrack():
    """Demo ByteTrack tracker."""
    print("ByteTrack Swimmer Tracker Demo")
    print("=" * 60)

    # Create tracker
    tracker = ByteTrackTracker(
        track_thresh=0.5,
        match_thresh=0.8,
    )

    print("✅ Tracker initialized")

    # Simulate detections over frames
    for frame_id in range(10):
        # Simulate moving swimmer
        x = 100 + frame_id * 10
        y = 200 + frame_id * 5

        detections = [
            {
                "bbox": np.array([x, y, x + 100, y + 150]),
                "keypoints": np.random.rand(17, 3) * 0.8 + 0.1,
                "confidence": 0.9,
            }
        ]

        # Update tracker
        tracks = tracker.update(detections, frame_id)

        print(f"\nFrame {frame_id}:")
        print(f"  Active tracks: {len(tracks)}")
        for track in tracks:
            print(
                f"    Track {track.track_id}: bbox={track.bbox}, hits={track.hits}, age={track.age}"
            )

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_bytetrack()
