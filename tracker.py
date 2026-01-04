"""
tracker.py - Computer Vision Module for BioTrack-Lite

This module implements the RodentTracker class which uses MOG2 background
subtraction to detect and track a single animal in a static arena.

Key Design Decisions:
- MOG2 is used because it's an explainable, classical CV algorithm that a
  student can explain in a research paper (unlike black-box deep learning).
- Shadow handling is critical because MOG2 marks shadows as gray (127),
  which would otherwise be detected as part of the animal.
- Morphological operations clean up sensor noise without losing the subject.

Body Part Detection:
- Head, body, and tail are estimated using contour geometry analysis
- Ellipse fitting determines the animal's orientation (major axis)
- Movement direction disambiguates head from tail (head leads movement)
- This is a geometric approximation, not true pose estimation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BodyParts:
    """Data class for storing detected body part positions."""
    head: Tuple[float, float]      # (x, y) of estimated head position
    body: Tuple[float, float]      # (x, y) of body center (centroid)
    tail: Tuple[float, float]      # (x, y) of estimated tail position
    orientation: float             # Angle in degrees (0-360)
    body_length: float             # Distance from head to tail in pixels
    confidence: float              # Detection confidence (0-1)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame storage."""
        return {
            'head_x': self.head[0],
            'head_y': self.head[1],
            'body_x': self.body[0],
            'body_y': self.body[1],
            'tail_x': self.tail[0],
            'tail_y': self.tail[1],
            'orientation': self.orientation,
            'body_length': self.body_length,
            'confidence': self.confidence
        }


class RodentTracker:
    """
    A computer vision tracker for single-animal behavioral analysis.

    Uses MOG2 (Mixture of Gaussians) background subtraction to segment
    the animal from a static arena background. Designed for offline
    analysis of pre-recorded Open Field Test videos.

    Attributes:
        history (int): Number of frames for background model (default: 500).
        var_threshold (int): MOG2 variance threshold - higher = less sensitive (default: 16).
        shadow_threshold (int): Binary threshold to remove shadows (default: 250).
        min_contour_area (int): Minimum contour size in pixels to consider valid (default: 500).
        blur_kernel (tuple): Gaussian blur kernel size for noise reduction (default: (21, 21)).

    Example:
        >>> tracker = RodentTracker(var_threshold=16, min_contour_area=500)
        >>> centroid, mask = tracker.process_frame(frame)
        >>> if not np.isnan(centroid[0]):
        ...     print(f"Animal at ({centroid[0]:.1f}, {centroid[1]:.1f})")
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 16,
        shadow_threshold: int = 250,
        min_contour_area: int = 500,
        blur_kernel: Tuple[int, int] = (21, 21)
    ):
        """
        Initialize the RodentTracker with configurable parameters.

        Args:
            history: Number of frames used to build the background model.
                     Higher values = more stable background, but slower adaptation.
            var_threshold: Mahalanobis distance threshold for MOG2.
                          Lower values = more sensitive detection (may pick up noise).
                          Higher values = less sensitive (may miss faint subjects).
            shadow_threshold: Binary threshold applied after MOG2.
                             MOG2 marks shadows as gray (127), foreground as white (255).
                             We threshold at 250 to keep only true foreground.
            min_contour_area: Minimum contour area in pixels to be considered valid.
                             Filters out noise blobs and small artifacts.
            blur_kernel: Gaussian blur kernel size (must be odd numbers).
                        Larger kernel = more smoothing = less noise but less detail.
        """
        self.history = history
        self.var_threshold = var_threshold
        self.shadow_threshold = shadow_threshold
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel

        # Initialize MOG2 background subtractor
        # detectShadows=True enables shadow detection (marked as gray in mask)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True
        )

        # Morphological structuring element for noise cleanup
        # Ellipse shape preserves curved edges better than rectangle
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Body part tracking state
        self.prev_centroid: Tuple[float, float] = (np.nan, np.nan)
        self.prev_head: Tuple[float, float] = (np.nan, np.nan)
        self.movement_history: list = []  # Store recent movement vectors
        self.history_size: int = 5  # Frames to average for direction

    def reset_background(self) -> None:
        """
        Reset the background model and body part tracking state.

        Call this when processing a new video to clear the learned background
        from the previous video.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True
        )
        # Reset body part tracking state
        self.prev_centroid = (np.nan, np.nan)
        self.prev_head = (np.nan, np.nan)
        self.movement_history = []

    def crop_roi(
        self,
        frame: np.ndarray,
        roi_pct: Dict[str, float]
    ) -> np.ndarray:
        """
        Crop the frame to a Region of Interest (ROI) based on percentage values.

        This is useful for removing cage edges, timestamps, or other artifacts
        that appear at the edges of the video frame.

        Args:
            frame: Input BGR image (H x W x 3).
            roi_pct: Dictionary with keys 'top', 'bottom', 'left', 'right'
                    specifying the percentage to crop from each edge.
                    Example: {'top': 5, 'bottom': 5, 'left': 10, 'right': 10}

        Returns:
            Cropped frame as numpy array.

        Why this matters:
            Many behavior videos have timestamps or cage bars visible at the edges.
            These artifacts can be misdetected as the animal, causing tracking errors.
        """
        h, w = frame.shape[:2]

        top = int(h * roi_pct.get('top', 0) / 100)
        bottom = int(h * (100 - roi_pct.get('bottom', 0)) / 100)
        left = int(w * roi_pct.get('left', 0) / 100)
        right = int(w * (100 - roi_pct.get('right', 0)) / 100)

        return frame[top:bottom, left:right]

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame before background subtraction.

        Applies Gaussian blur to reduce high-frequency noise from the camera
        sensor (thermal noise, compression artifacts).

        Args:
            frame: Input BGR image.

        Returns:
            Blurred frame.

        Why Gaussian blur:
            Camera sensors produce random pixel-level noise ("thermal noise").
            Without blurring, this noise would appear as flickering pixels in the
            foreground mask, making contour detection unreliable.
            A 21x21 kernel is large enough to smooth noise but small enough to
            preserve the animal's shape.
        """
        return cv2.GaussianBlur(frame, self.blur_kernel, 0)

    def apply_mog2(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply MOG2 background subtraction with shadow removal.

        MOG2 (Mixture of Gaussians 2) models each pixel as a mixture of
        Gaussian distributions. Pixels that don't fit the background model
        are marked as foreground.

        Args:
            frame: Preprocessed (blurred) BGR image.

        Returns:
            Binary mask where 255 = foreground (animal), 0 = background.

        Shadow Handling (Critical for accuracy):
            MOG2 with detectShadows=True marks shadows as gray (127) and
            foreground as white (255). Without explicit shadow removal,
            the animal's shadow would be detected as part of its body,
            causing the centroid to shift toward the shadow.

            We apply a binary threshold at 250 to eliminate shadows:
            - Pixels > 250 -> 255 (true foreground)
            - Pixels <= 250 -> 0 (background + shadows)
        """
        # Apply background subtraction
        mask = self.bg_subtractor.apply(frame)

        # Remove shadows by thresholding
        # MOG2 marks: foreground=255, shadows=127, background=0
        # We only want true foreground (255), so threshold at 250
        _, mask = cv2.threshold(
            mask,
            self.shadow_threshold,
            255,
            cv2.THRESH_BINARY
        )

        return mask

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean the binary mask using morphological operations.

        Applies dilation followed by erosion ("closing" variant) to:
        1. Fill small holes in the detected animal
        2. Remove salt-and-pepper noise (small bright spots)

        Args:
            mask: Binary foreground mask from apply_mog2().

        Returns:
            Cleaned binary mask.

        Why Dilation then Erosion:
            - Dilation expands white regions, filling small gaps in the animal
            - Erosion shrinks white regions, removing isolated noise pixels
            - The order matters: dilation first fills holes, then erosion
              removes the expansion while keeping the filled holes

            This is similar to cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            but done in two steps for clarity.
        """
        # Dilation: expand white regions (fills holes in the animal)
        mask = cv2.dilate(mask, self.morph_kernel, iterations=2)

        # Erosion: shrink white regions (removes isolated noise)
        mask = cv2.erode(mask, self.morph_kernel, iterations=2)

        return mask

    def find_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        Find the centroid of the largest valid contour in the mask.

        Implements the single-subject assumption: always select the LARGEST
        contour that meets the minimum area threshold.

        Args:
            mask: Cleaned binary mask.

        Returns:
            Tuple (x, y) of centroid coordinates in pixels.
            Returns (np.nan, np.nan) if no valid contour is found.

        Error Handling Philosophy:
            When no contour is found, we return NaN rather than interpolating
            or returning the last known position. This is scientifically honest:
            - NaN values in the output data clearly indicate tracking failures
            - Downstream analysis can handle NaN appropriately (skip or flag)
            - The user can see exactly which frames had detection issues

            Silent interpolation would hide tracking problems and could lead
            to incorrect scientific conclusions.
        """
        # Find all contours in the mask
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return (np.nan, np.nan)

        # Filter contours by minimum area
        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) >= self.min_contour_area
        ]

        if not valid_contours:
            return (np.nan, np.nan)

        # Select the largest contour (single-subject assumption)
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Calculate centroid using image moments
        moments = cv2.moments(largest_contour)

        # m00 is the area; if zero, contour is degenerate
        if moments['m00'] == 0:
            return (np.nan, np.nan)

        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

        return (cx, cy)

    def find_body_parts(
        self,
        mask: np.ndarray,
        centroid: Tuple[float, float]
    ) -> Optional[BodyParts]:
        """
        Estimate head, body, and tail positions from contour geometry.

        Uses ellipse fitting and movement direction to estimate body parts.
        The head is determined by the direction of movement (animals move head-first).

        Args:
            mask: Cleaned binary mask.
            centroid: Pre-calculated centroid from find_centroid().

        Returns:
            BodyParts object with estimated positions, or None if detection fails.

        Algorithm:
            1. Find the largest contour (same as centroid detection)
            2. Fit an ellipse to get orientation and major/minor axes
            3. Find the two extreme points along the major axis
            4. Use movement history to determine which end is the head
            5. If stationary, use proximity to previous head position
        """
        if np.isnan(centroid[0]):
            return None

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Filter and get largest contour
        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) >= self.min_contour_area
        ]

        if not valid_contours:
            return None

        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Need at least 5 points for ellipse fitting
        if len(largest_contour) < 5:
            return None

        # Fit ellipse to contour
        try:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (minor_axis, major_axis), angle = ellipse
        except cv2.error:
            return None

        # Calculate endpoints along the major axis
        # Angle is in degrees, convert to radians
        angle_rad = np.radians(angle)

        # The semi-major axis length
        semi_major = major_axis / 2

        # Calculate the two endpoints along the major axis
        # Note: OpenCV ellipse angle is from the horizontal axis
        dx = semi_major * np.cos(angle_rad - np.pi/2)
        dy = semi_major * np.sin(angle_rad - np.pi/2)

        endpoint1 = (centroid[0] + dx, centroid[1] + dy)
        endpoint2 = (centroid[0] - dx, centroid[1] - dy)

        # Determine which endpoint is the head using movement direction
        head, tail = self._determine_head_tail(endpoint1, endpoint2, centroid)

        # Calculate body length
        body_length = np.sqrt(
            (head[0] - tail[0])**2 + (head[1] - tail[1])**2
        )

        # Calculate orientation (angle from tail to head)
        orientation = np.degrees(np.arctan2(
            head[1] - tail[1],
            head[0] - tail[0]
        ))
        if orientation < 0:
            orientation += 360

        # Confidence based on ellipse elongation (more elongated = more confident)
        elongation = major_axis / minor_axis if minor_axis > 0 else 1
        confidence = min(1.0, (elongation - 1) / 3)  # Max confidence at 4:1 ratio

        # Update tracking state
        self.prev_head = head

        return BodyParts(
            head=head,
            body=centroid,
            tail=tail,
            orientation=orientation,
            body_length=body_length,
            confidence=confidence
        )

    def _determine_head_tail(
        self,
        endpoint1: Tuple[float, float],
        endpoint2: Tuple[float, float],
        centroid: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Determine which endpoint is the head and which is the tail.

        Uses movement direction as the primary indicator (head leads movement).
        Falls back to previous head position if stationary.

        Args:
            endpoint1: First endpoint along major axis.
            endpoint2: Second endpoint along major axis.
            centroid: Current body center.

        Returns:
            Tuple of (head, tail) positions.
        """
        # Update movement history
        if not np.isnan(self.prev_centroid[0]):
            dx = centroid[0] - self.prev_centroid[0]
            dy = centroid[1] - self.prev_centroid[1]
            movement = np.sqrt(dx**2 + dy**2)

            if movement > 2:  # Minimum movement threshold (pixels)
                self.movement_history.append((dx, dy))
                if len(self.movement_history) > self.history_size:
                    self.movement_history.pop(0)

        self.prev_centroid = centroid

        # Calculate average movement direction
        if len(self.movement_history) >= 2:
            avg_dx = np.mean([m[0] for m in self.movement_history])
            avg_dy = np.mean([m[1] for m in self.movement_history])

            # Dot product to determine which endpoint is in movement direction
            dot1 = (endpoint1[0] - centroid[0]) * avg_dx + \
                   (endpoint1[1] - centroid[1]) * avg_dy
            dot2 = (endpoint2[0] - centroid[0]) * avg_dx + \
                   (endpoint2[1] - centroid[1]) * avg_dy

            if dot1 > dot2:
                return endpoint1, endpoint2
            else:
                return endpoint2, endpoint1

        # Fallback: use proximity to previous head position
        if not np.isnan(self.prev_head[0]):
            dist1 = np.sqrt(
                (endpoint1[0] - self.prev_head[0])**2 +
                (endpoint1[1] - self.prev_head[1])**2
            )
            dist2 = np.sqrt(
                (endpoint2[0] - self.prev_head[0])**2 +
                (endpoint2[1] - self.prev_head[1])**2
            )

            if dist1 < dist2:
                return endpoint1, endpoint2
            else:
                return endpoint2, endpoint1

        # Default: arbitrary assignment (first frame, stationary)
        return endpoint1, endpoint2

    def process_frame(
        self,
        frame: np.ndarray,
        roi_pct: Optional[Dict[str, float]] = None,
        detect_body_parts: bool = False
    ) -> Tuple[Tuple[float, float], np.ndarray, Optional[BodyParts]]:
        """
        Process a single video frame through the complete tracking pipeline.

        This is the main entry point for frame-by-frame tracking.

        Pipeline:
            1. Crop to ROI (if specified)
            2. Preprocess (Gaussian blur)
            3. Background subtraction (MOG2)
            4. Shadow removal (binary threshold)
            5. Morphological cleanup
            6. Contour detection
            7. Centroid calculation
            8. Body part estimation (optional)

        Args:
            frame: Input BGR image from video.
            roi_pct: Optional ROI crop percentages. If None, uses full frame.
            detect_body_parts: If True, also estimate head/body/tail positions.

        Returns:
            Tuple of:
                - (x, y): Centroid coordinates (or (np.nan, np.nan) if not found)
                - mask: Processed binary mask for visualization
                - body_parts: BodyParts object if detect_body_parts=True, else None

        Example:
            >>> cap = cv2.VideoCapture('mouse_video.mp4')
            >>> tracker = RodentTracker()
            >>> ret, frame = cap.read()
            >>> centroid, mask, body_parts = tracker.process_frame(frame, detect_body_parts=True)
        """
        # Step 1: Crop to ROI if specified
        if roi_pct is not None:
            frame = self.crop_roi(frame, roi_pct)

        # Step 2: Preprocess (reduce noise)
        blurred = self.preprocess(frame)

        # Step 3 & 4: Background subtraction with shadow removal
        mask = self.apply_mog2(blurred)

        # Step 5: Morphological cleanup
        mask = self.clean_mask(mask)

        # Step 6 & 7: Find centroid
        centroid = self.find_centroid(mask)

        # Step 8: Body part estimation (optional)
        body_parts = None
        if detect_body_parts:
            body_parts = self.find_body_parts(mask, centroid)

        return centroid, mask, body_parts

    def annotate_frame(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
        roi_pct: Optional[Dict[str, float]] = None,
        body_parts: Optional[BodyParts] = None
    ) -> np.ndarray:
        """
        Draw the detected centroid and body parts on the frame for visualization.

        Args:
            frame: Original BGR frame.
            centroid: (x, y) coordinates from process_frame().
            roi_pct: ROI percentages (needed to offset coordinates correctly).
            body_parts: Optional BodyParts object for head/body/tail visualization.

        Returns:
            Annotated frame with tracking markers.

        Color Scheme:
            - Head: Green (0, 255, 0)
            - Body: Yellow (0, 255, 255)
            - Tail: Red (0, 0, 255)
            - Skeleton line: Cyan (255, 255, 0)
        """
        annotated = frame.copy()

        # Calculate offset if ROI cropping was applied
        h, w = frame.shape[:2]
        x_offset = int(w * roi_pct.get('left', 0) / 100) if roi_pct else 0
        y_offset = int(h * roi_pct.get('top', 0) / 100) if roi_pct else 0

        if body_parts is not None:
            # Draw body parts with skeleton
            head = (int(body_parts.head[0]) + x_offset,
                    int(body_parts.head[1]) + y_offset)
            body = (int(body_parts.body[0]) + x_offset,
                    int(body_parts.body[1]) + y_offset)
            tail = (int(body_parts.tail[0]) + x_offset,
                    int(body_parts.tail[1]) + y_offset)

            # Draw skeleton line (tail -> body -> head)
            cv2.line(annotated, tail, body, (255, 255, 0), 2)
            cv2.line(annotated, body, head, (255, 255, 0), 2)

            # Draw tail (red)
            cv2.circle(annotated, tail, 6, (255, 255, 255), 2)
            cv2.circle(annotated, tail, 4, (0, 0, 255), -1)

            # Draw body center (yellow)
            cv2.circle(annotated, body, 8, (255, 255, 255), 2)
            cv2.circle(annotated, body, 5, (0, 255, 255), -1)

            # Draw head (green) - larger to emphasize
            cv2.circle(annotated, head, 10, (255, 255, 255), 2)
            cv2.circle(annotated, head, 7, (0, 255, 0), -1)

            # Draw orientation indicator (arrow from body to head)
            arrow_len = 20
            angle_rad = np.radians(body_parts.orientation)
            arrow_end = (
                int(head[0] + arrow_len * np.cos(angle_rad)),
                int(head[1] + arrow_len * np.sin(angle_rad))
            )
            cv2.arrowedLine(annotated, head, arrow_end, (0, 255, 0), 2, tipLength=0.4)

            # Add label with confidence
            label = f"Conf: {body_parts.confidence:.2f}"
            cv2.putText(annotated, label, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif not np.isnan(centroid[0]):
            # Fallback: just draw centroid if no body parts detected
            cx = int(centroid[0]) + x_offset
            cy = int(centroid[1]) + y_offset

            # Outer circle (white border)
            cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 2)
            # Inner circle (red fill)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        return annotated
