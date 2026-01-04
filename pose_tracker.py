"""
pose_tracker.py - YOLOv8 Pose Estimation Module for BioTrack-Lite

This module implements pose estimation using YOLOv8-Nano Pose model
for single-animal tracking with keypoint detection.

Key Features:
- Automatic model download on first use (~6MB)
- CPU-friendly inference for frugal lab setups
- Fallback to centroid when pose detection fails
- Skeleton rendering for visualization
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from ultralytics import YOLO


class PoseTracker:
    """
    YOLOv8-based pose estimation tracker for behavioral analysis.

    Uses YOLOv8-Nano Pose model for keypoint detection. When pose
    estimation fails, falls back to bounding box centroid or returns NaN.

    Keypoint Mapping (COCO format adapted for rodents):
        0: nose
        5, 6: shoulders (interpreted as forepaws)
        11, 12: hips (interpreted as hindpaws)
        Other keypoints available but may be less reliable on quadrupeds

    Attributes:
        model: YOLOv8 Pose model instance
        conf_threshold: Minimum confidence for valid detection
        device: Inference device ('cpu' or 'cuda:0')

    Example:
        >>> tracker = PoseTracker()
        >>> pose_data, annotated = tracker.process_frame(frame)
        >>> if pose_data['keypoints'] is not None:
        ...     print(f"Nose at ({pose_data['keypoints'][0, 0]:.1f}, {pose_data['keypoints'][0, 1]:.1f})")
    """

    # Skeleton connections for drawing (simplified for rodent body)
    SKELETON_CONNECTIONS = [
        (0, 5), (0, 6),    # nose to shoulders (forepaws)
        (5, 6),            # shoulder line
        (5, 11), (6, 12),  # body sides
        (11, 12),          # hip line
    ]

    # Keypoint colors (BGR format)
    KEYPOINT_COLORS = {
        0: (0, 255, 255),    # nose - yellow
        5: (255, 0, 255),    # left shoulder - magenta
        6: (255, 0, 255),    # right shoulder - magenta
        11: (0, 255, 0),     # left hip - green
        12: (0, 255, 0),     # right hip - green
    }

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        conf_threshold: float = 0.3,
        device: str = "cpu"
    ):
        """
        Initialize the PoseTracker with YOLOv8 Pose model.

        Args:
            model_path: Path to YOLOv8 pose model weights.
                       Will auto-download if not found.
            conf_threshold: Minimum confidence threshold for detections.
                           Lower = more detections but more false positives.
            device: Inference device ('cpu' or 'cuda:0' for GPU).
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

        # Track previous centroid for velocity calculation
        self.prev_centroid = None

    def crop_roi(
        self,
        frame: np.ndarray,
        roi_pct: Dict[str, float]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop the frame to Region of Interest.

        Args:
            frame: Input BGR image.
            roi_pct: Dictionary with 'top', 'bottom', 'left', 'right' percentages.

        Returns:
            Tuple of (cropped_frame, (x_offset, y_offset))
        """
        h, w = frame.shape[:2]

        top = int(h * roi_pct.get('top', 0) / 100)
        bottom = int(h * (100 - roi_pct.get('bottom', 0)) / 100)
        left = int(w * roi_pct.get('left', 0) / 100)
        right = int(w * (100 - roi_pct.get('right', 0)) / 100)

        return frame[top:bottom, left:right], (left, top)

    def process_frame(
        self,
        frame: np.ndarray,
        roi_pct: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict, np.ndarray]:
        """
        Process a single frame through YOLOv8 Pose estimation.

        Pipeline:
            1. Crop to ROI (if specified)
            2. Run YOLOv8 pose inference
            3. Extract keypoints and centroid
            4. Draw skeleton overlay

        Args:
            frame: Input BGR image from video.
            roi_pct: Optional ROI crop percentages.

        Returns:
            Tuple of:
                - pose_data: Dictionary containing:
                    - 'keypoints': (17, 3) array [x, y, confidence] or None
                    - 'centroid': (x, y) center of detection
                    - 'bbox': [x1, y1, x2, y2] bounding box or None
                    - 'confidence': overall detection confidence
                    - 'fallback': True if using bbox centroid instead of pose
                - annotated_frame: Frame with pose skeleton drawn
        """
        # Apply ROI cropping if specified
        offset = (0, 0)
        if roi_pct is not None:
            frame, offset = self.crop_roi(frame, roi_pct)

        # Run inference
        results = self.model(frame, verbose=False, device=self.device)
        result = results[0]

        # Initialize output
        pose_data = {
            'keypoints': None,
            'centroid': (np.nan, np.nan),
            'bbox': None,
            'confidence': 0.0,
            'fallback': True,
            'body_length': np.nan,
            'body_angle': np.nan
        }

        # Check if any detections
        if len(result.boxes) == 0:
            return pose_data, frame

        # Get the highest confidence detection
        confidences = result.boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        best_conf = confidences[best_idx]

        if best_conf < self.conf_threshold:
            return pose_data, frame

        # Extract bounding box
        bbox = result.boxes.xyxy[best_idx].cpu().numpy()
        pose_data['bbox'] = bbox
        pose_data['confidence'] = float(best_conf)

        # Calculate bbox centroid as fallback
        bbox_centroid = (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        )

        # Extract keypoints if available
        if result.keypoints is not None and len(result.keypoints) > 0:
            kpts = result.keypoints[best_idx].data.cpu().numpy()[0]  # (17, 3)

            # Check if keypoints are valid (not all zeros)
            if np.any(kpts[:, :2] > 0):
                pose_data['keypoints'] = kpts
                pose_data['fallback'] = False

                # Calculate centroid from valid keypoints
                valid_kpts = kpts[kpts[:, 2] > 0.3]  # confidence > 0.3
                if len(valid_kpts) > 0:
                    pose_data['centroid'] = (
                        float(np.mean(valid_kpts[:, 0])),
                        float(np.mean(valid_kpts[:, 1]))
                    )
                else:
                    pose_data['centroid'] = bbox_centroid

                # Calculate body length (nose to hip center)
                pose_data['body_length'] = self._calculate_body_length(kpts)
                pose_data['body_angle'] = self._calculate_body_angle(kpts)
            else:
                pose_data['centroid'] = bbox_centroid
        else:
            pose_data['centroid'] = bbox_centroid

        # Draw skeleton overlay
        annotated = self.draw_skeleton(frame, pose_data)

        return pose_data, annotated

    def _calculate_body_length(self, keypoints: np.ndarray) -> float:
        """Calculate body length from nose to hip center."""
        nose = keypoints[0, :2]
        hip_center = np.mean([keypoints[11, :2], keypoints[12, :2]], axis=0)

        if keypoints[0, 2] < 0.3 or (keypoints[11, 2] < 0.3 and keypoints[12, 2] < 0.3):
            return np.nan

        return float(np.linalg.norm(nose - hip_center))

    def _calculate_body_angle(self, keypoints: np.ndarray) -> float:
        """Calculate body angle (degrees from horizontal)."""
        nose = keypoints[0, :2]
        hip_center = np.mean([keypoints[11, :2], keypoints[12, :2]], axis=0)

        if keypoints[0, 2] < 0.3 or (keypoints[11, 2] < 0.3 and keypoints[12, 2] < 0.3):
            return np.nan

        delta = hip_center - nose
        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        return float(angle)

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose_data: Dict,
        skeleton_color: Tuple[int, int, int] = (0, 255, 0),
        keypoint_radius: int = 4,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw pose skeleton overlay on frame.

        Args:
            frame: Input BGR image.
            pose_data: Dictionary from process_frame().
            skeleton_color: BGR color for skeleton lines.
            keypoint_radius: Radius of keypoint circles.
            line_thickness: Thickness of skeleton lines.

        Returns:
            Annotated frame with skeleton drawn.
        """
        annotated = frame.copy()
        keypoints = pose_data.get('keypoints')

        if keypoints is None:
            # Draw bbox centroid only
            centroid = pose_data.get('centroid')
            if centroid is not None and not np.isnan(centroid[0]):
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 2)
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            # Draw bounding box if available
            bbox = pose_data.get('bbox')
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

            return annotated

        # Draw skeleton connections
        for (i, j) in self.SKELETON_CONNECTIONS:
            if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
                pt1 = tuple(keypoints[i, :2].astype(int))
                pt2 = tuple(keypoints[j, :2].astype(int))
                cv2.line(annotated, pt1, pt2, skeleton_color, line_thickness)

        # Draw keypoints
        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0.3:
                pt = tuple(keypoints[i, :2].astype(int))
                color = self.KEYPOINT_COLORS.get(i, (0, 255, 0))
                cv2.circle(annotated, pt, keypoint_radius, color, -1)
                cv2.circle(annotated, pt, keypoint_radius + 2, (255, 255, 255), 1)

        # Draw centroid
        centroid = pose_data.get('centroid')
        if centroid is not None and not np.isnan(centroid[0]):
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.drawMarker(
                annotated, (cx, cy),
                color=(0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=15,
                thickness=2
            )

        # Draw confidence text
        conf = pose_data.get('confidence', 0)
        cv2.putText(
            annotated,
            f"Conf: {conf:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return annotated

    def extract_pose_features(self, pose_data: Dict) -> Dict:
        """
        Extract geometric features from pose keypoints for behavior classification.

        Args:
            pose_data: Dictionary from process_frame().

        Returns:
            Dictionary of pose features:
                - nose_height: Y position of nose (lower = higher in frame)
                - forepaw_height: Average Y of forepaws
                - hindpaw_height: Average Y of hindpaws
                - forepaw_elevation: Difference between hindpaw and forepaw height
                - body_length: Distance from nose to hip center
                - body_angle: Angle of body axis (degrees from horizontal)
                - is_compact: True if body length < threshold (grooming posture)
        """
        features = {
            'nose_height': np.nan,
            'forepaw_height': np.nan,
            'hindpaw_height': np.nan,
            'forepaw_elevation': np.nan,
            'body_length': pose_data.get('body_length', np.nan),
            'body_angle': pose_data.get('body_angle', np.nan),
            'is_compact': False,
            'nose_x': np.nan,
            'nose_y': np.nan
        }

        keypoints = pose_data.get('keypoints')
        if keypoints is None:
            return features

        # Nose position
        if keypoints[0, 2] > 0.3:
            features['nose_height'] = keypoints[0, 1]
            features['nose_x'] = keypoints[0, 0]
            features['nose_y'] = keypoints[0, 1]

        # Forepaw height (shoulders in COCO)
        forepaw_heights = []
        if keypoints[5, 2] > 0.3:
            forepaw_heights.append(keypoints[5, 1])
        if keypoints[6, 2] > 0.3:
            forepaw_heights.append(keypoints[6, 1])
        if forepaw_heights:
            features['forepaw_height'] = np.mean(forepaw_heights)

        # Hindpaw height (hips in COCO)
        hindpaw_heights = []
        if keypoints[11, 2] > 0.3:
            hindpaw_heights.append(keypoints[11, 1])
        if keypoints[12, 2] > 0.3:
            hindpaw_heights.append(keypoints[12, 1])
        if hindpaw_heights:
            features['hindpaw_height'] = np.mean(hindpaw_heights)

        # Forepaw elevation (positive = forepaws above hindpaws = rearing)
        if not np.isnan(features['forepaw_height']) and not np.isnan(features['hindpaw_height']):
            # In image coordinates, lower Y = higher position
            features['forepaw_elevation'] = features['hindpaw_height'] - features['forepaw_height']

        # Compact body detection (for grooming)
        body_length = features['body_length']
        if not np.isnan(body_length):
            # Compact if body length < 60% of typical extended length
            # This threshold should be calibrated per setup
            features['is_compact'] = body_length < 50  # pixels, adjustable

        return features

    def reset(self):
        """Reset tracker state for new video."""
        self.prev_centroid = None
