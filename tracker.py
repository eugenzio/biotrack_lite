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
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


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

    def reset_background(self) -> None:
        """
        Reset the background model.

        Call this when processing a new video to clear the learned background
        from the previous video.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True
        )

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

    def process_frame(
        self,
        frame: np.ndarray,
        roi_pct: Optional[Dict[str, float]] = None
    ) -> Tuple[Tuple[float, float], np.ndarray]:
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

        Args:
            frame: Input BGR image from video.
            roi_pct: Optional ROI crop percentages. If None, uses full frame.

        Returns:
            Tuple of:
                - (x, y): Centroid coordinates (or (np.nan, np.nan) if not found)
                - mask: Processed binary mask for visualization

        Example:
            >>> cap = cv2.VideoCapture('mouse_video.mp4')
            >>> tracker = RodentTracker()
            >>> ret, frame = cap.read()
            >>> centroid, mask = tracker.process_frame(frame)
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

        return centroid, mask

    def annotate_frame(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
        roi_pct: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Draw the detected centroid on the frame for visualization.

        Args:
            frame: Original BGR frame.
            centroid: (x, y) coordinates from process_frame().
            roi_pct: ROI percentages (needed to offset coordinates correctly).

        Returns:
            Annotated frame with centroid marker.
        """
        annotated = frame.copy()

        if not np.isnan(centroid[0]):
            # Calculate offset if ROI cropping was applied
            h, w = frame.shape[:2]
            x_offset = int(w * roi_pct.get('left', 0) / 100) if roi_pct else 0
            y_offset = int(h * roi_pct.get('top', 0) / 100) if roi_pct else 0

            # Draw centroid marker
            cx = int(centroid[0]) + x_offset
            cy = int(centroid[1]) + y_offset

            # Outer circle (white border)
            cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 2)
            # Inner circle (red fill)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        return annotated
