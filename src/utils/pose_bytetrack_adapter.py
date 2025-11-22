"""
Adapter to convert between pose estimator outputs and ByteTrack detection format.

Handles format differences:
- Pose estimators return: list[dict] with 'keypoints', 'bbox', 'confidence'
- ByteTrack expects: list[dict] with 'bbox' as [x1, y1, x2, y2], 'score', 'class_id'
"""

from typing import Any

import numpy as np


def pose_to_bytetrack_detection(pose_data: dict) -> dict[str, Any]:
    """
    Convert single pose estimator output to ByteTrack detection format.

    Args:
        pose_data: Pose dictionary from estimator with format:
            {
                'keypoints': np.ndarray (N, 3),
                'bbox': list[float] or dict with x1,y1,x2,y2,
                'confidence': float,
                'person_id': int,
                ...
            }

    Returns:
        ByteTrack detection dictionary:
            {
                'bbox': [x1, y1, x2, y2],  # List format
                'score': float,
                'class_id': int,  # Always 0 for person
                'keypoints': np.ndarray,  # Preserved for downstream use
            }
    """
    bbox = pose_data.get("bbox")

    # Handle different bbox formats
    if bbox is None:
        # If no bbox, estimate from keypoints
        keypoints = pose_data.get("keypoints")
        if keypoints is not None and len(keypoints) > 0:
            if isinstance(keypoints, np.ndarray):
                valid_kpts = keypoints[keypoints[:, 2] > 0]  # Filter by confidence
                if len(valid_kpts) > 0:
                    x_coords = valid_kpts[:, 0]
                    y_coords = valid_kpts[:, 1]
                    x1, y1 = x_coords.min(), y_coords.min()
                    x2, y2 = x_coords.max(), y_coords.max()
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
    elif isinstance(bbox, dict):
        # Convert dict format to list
        bbox = [
            float(bbox.get("x1", 0)),
            float(bbox.get("y1", 0)),
            float(bbox.get("x2", 0)),
            float(bbox.get("y2", 0)),
        ]
    elif isinstance(bbox, (list, tuple)):
        # Ensure it's a list of floats
        bbox = [float(x) for x in bbox[:4]]
    else:
        bbox = [0.0, 0.0, 0.0, 0.0]

    # Get confidence/score
    confidence = pose_data.get("confidence", 0.0)
    if isinstance(confidence, (np.ndarray, np.generic)):
        confidence = float(confidence)

    # Build ByteTrack detection
    detection = {
        "bbox": bbox,
        "score": confidence,
        "class_id": 0,  # Person class
        "keypoints": pose_data.get("keypoints"),  # Preserve keypoints
        "person_id": pose_data.get("person_id", -1),  # Preserve if exists
    }

    return detection


def poses_to_bytetrack_detections(pose_data_list: list[dict] | None) -> list[dict]:
    """
    Convert list of pose estimator outputs to ByteTrack detection format.

    Args:
        pose_data_list: List of pose dictionaries from estimator, or None

    Returns:
        List of ByteTrack detection dictionaries
    """
    if pose_data_list is None or len(pose_data_list) == 0:
        return []

    return [pose_to_bytetrack_detection(pose) for pose in pose_data_list]


def bytetrack_to_pose_data(detection: dict, keypoint_format: Any) -> dict:
    """
    Convert ByteTrack detection back to pose data format.

    Useful after tracking to reconstruct pose data with track IDs.

    Args:
        detection: ByteTrack detection with track_id
        keypoint_format: KeypointFormat enum value

    Returns:
        Pose data dictionary
    """
    bbox_list = detection.get("bbox", [0, 0, 0, 0])

    pose_data = {
        "keypoints": detection.get("keypoints"),
        "bbox": bbox_list if isinstance(bbox_list, list) else list(bbox_list),
        "confidence": detection.get("score", 0.0),
        "person_id": detection.get("track_id", detection.get("person_id", -1)),
        "format": keypoint_format,
        "metadata": {
            "tracked": True,
            "track_id": detection.get("track_id", -1),
        },
    }

    return pose_data


# Example usage
if __name__ == "__main__":
    # Test conversion
    import numpy as np

    from src.pose.base_estimator import KeypointFormat

    # Sample pose data from estimator
    pose_sample = {
        "keypoints": np.random.rand(17, 3),
        "bbox": [100, 100, 300, 400],
        "confidence": 0.85,
        "person_id": 0,
        "format": KeypointFormat.COCO_17,
    }

    # Convert to ByteTrack format
    detection = pose_to_bytetrack_detection(pose_sample)
    print("ByteTrack detection:")
    print(f"  bbox: {detection['bbox']}")
    print(f"  score: {detection['score']}")
    print(f"  class_id: {detection['class_id']}")

    # Convert back
    pose_reconstructed = bytetrack_to_pose_data(detection, KeypointFormat.COCO_17)
    print("\nReconstructed pose:")
    print(f"  bbox: {pose_reconstructed['bbox']}")
    print(f"  confidence: {pose_reconstructed['confidence']}")
    print(f"  person_id: {pose_reconstructed['person_id']}")
