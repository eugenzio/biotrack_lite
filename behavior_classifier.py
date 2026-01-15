"""
behavior_classifier.py - Rule-Based Behavior Classification for Movyent

This module implements behavior detection using interpretable rules based on
pose geometry and velocity. No machine learning training required.

Detected Behaviors:
    - locomotion: Active movement (velocity > threshold)
    - rearing: Forepaws elevated, vertical posture
    - grooming: Compact posture, forepaws near head
    - freezing: Immobility for minimum duration
    - wall_sniffing: Nose near arena wall
    - center_exploration: Moving in center zone

Design Philosophy:
    - All rules are interpretable and can be described in a research paper
    - Thresholds are exposed for per-setup calibration
    - Temporal behaviors (freezing) use state tracking for duration requirements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BehaviorThresholds:
    """Configuration for behavior detection thresholds."""
    # Locomotion
    locomotion_velocity_min: float = 5.0  # cm/s - above this = locomotion

    # Freezing
    freezing_velocity_max: float = 2.0  # cm/s - below this = immobile
    freezing_min_duration: float = 1.0  # seconds

    # Rearing
    rearing_elevation_min: float = 20.0  # pixels - forepaw elevation threshold

    # Grooming
    grooming_body_compact_ratio: float = 0.6  # body length ratio threshold
    grooming_velocity_max: float = 3.0  # cm/s

    # Wall-sniffing
    wall_distance_max: float = 3.0  # cm from wall

    # Center exploration
    center_zone_pct: float = 0.6  # fraction of arena considered center


class BehaviorStateTracker:
    """
    Track temporal state for duration-dependent behaviors.

    Implements a simple state machine for behaviors that require
    minimum duration (e.g., freezing must last > 1 second).
    """

    def __init__(self, fps: float, min_freezing_duration: float = 1.0):
        """
        Initialize state tracker.

        Args:
            fps: Video frame rate for duration calculation.
            min_freezing_duration: Minimum seconds for confirmed freezing.
        """
        self.fps = fps
        self.min_freezing_frames = int(min_freezing_duration * fps)

        # State tracking
        self.immobile_counter = 0
        self.confirmed_freezing = False

    def update(self, is_immobile: bool) -> str:
        """
        Update freezing state machine.

        Args:
            is_immobile: True if velocity below threshold this frame.

        Returns:
            'freezing' if confirmed, 'freezing_candidate' if building up,
            'active' if moving.
        """
        if is_immobile:
            self.immobile_counter += 1
            if self.immobile_counter >= self.min_freezing_frames:
                self.confirmed_freezing = True
                return 'freezing'
            return 'freezing_candidate'
        else:
            self.immobile_counter = 0
            self.confirmed_freezing = False
            return 'active'

    def reset(self):
        """Reset state for new video."""
        self.immobile_counter = 0
        self.confirmed_freezing = False


class BehaviorClassifier:
    """
    Rule-based behavior classification from pose and velocity data.

    Classifies frame-by-frame behaviors using interpretable rules.
    Multiple behaviors can be active simultaneously (e.g., locomotion + wall_sniffing).

    Example:
        >>> classifier = BehaviorClassifier(arena_width=400, arena_height=400,
        ...                                  pixels_per_cm=10, fps=30)
        >>> behaviors = classifier.classify_frame(pose_data, pose_features, velocity)
        >>> print(behaviors)  # ['locomotion', 'wall_sniffing']
    """

    def __init__(
        self,
        arena_width: float,
        arena_height: float,
        pixels_per_cm: float,
        fps: float,
        thresholds: Optional[BehaviorThresholds] = None
    ):
        """
        Initialize behavior classifier.

        Args:
            arena_width: Arena width in pixels.
            arena_height: Arena height in pixels.
            pixels_per_cm: Calibration factor for distance measurements.
            fps: Video frame rate.
            thresholds: Optional custom thresholds. Uses defaults if None.
        """
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.pixels_per_cm = pixels_per_cm
        self.fps = fps
        self.thresholds = thresholds or BehaviorThresholds()

        # State tracker for temporal behaviors
        self.state_tracker = BehaviorStateTracker(
            fps=fps,
            min_freezing_duration=self.thresholds.freezing_min_duration
        )

        # Store reference body length for compact detection
        self.reference_body_length: Optional[float] = None

    def classify_frame(
        self,
        pose_data: Dict,
        pose_features: Dict,
        velocity: float
    ) -> List[str]:
        """
        Classify behaviors for a single frame.

        Args:
            pose_data: Dictionary from PoseTracker.process_frame().
            pose_features: Dictionary from PoseTracker.extract_pose_features().
            velocity: Current velocity in cm/s.

        Returns:
            List of active behavior names. Can be empty or contain multiple.
        """
        behaviors = []

        # Handle NaN velocity
        if np.isnan(velocity):
            return behaviors

        # 1. Locomotion: High velocity movement
        if self._detect_locomotion(velocity):
            behaviors.append('locomotion')

        # 2. Freezing: Low velocity for extended duration
        freezing_state = self._detect_freezing(velocity)
        if freezing_state == 'freezing':
            behaviors.append('freezing')

        # 3. Rearing: Forepaws elevated
        if self._detect_rearing(pose_features):
            behaviors.append('rearing')

        # 4. Grooming: Compact posture, low velocity
        if self._detect_grooming(pose_features, velocity):
            behaviors.append('grooming')

        # 5. Wall-sniffing: Nose near wall
        if self._detect_wall_sniffing(pose_data, pose_features):
            behaviors.append('wall_sniffing')

        # 6. Center exploration: In center zone and moving
        if self._detect_center_exploration(pose_data, velocity):
            behaviors.append('center_exploration')

        return behaviors

    def _detect_locomotion(self, velocity: float) -> bool:
        """Detect locomotion based on velocity threshold."""
        return velocity > self.thresholds.locomotion_velocity_min

    def _detect_freezing(self, velocity: float) -> str:
        """
        Detect freezing behavior with duration requirement.

        Returns state string for tracking, 'freezing' only when confirmed.
        """
        is_immobile = velocity < self.thresholds.freezing_velocity_max
        return self.state_tracker.update(is_immobile)

    def _detect_rearing(self, pose_features: Dict) -> bool:
        """
        Detect rearing behavior from pose geometry.

        Rearing = forepaws significantly elevated above hindpaws.
        """
        elevation = pose_features.get('forepaw_elevation', np.nan)

        if np.isnan(elevation):
            return False

        return elevation > self.thresholds.rearing_elevation_min

    def _detect_grooming(self, pose_features: Dict, velocity: float) -> bool:
        """
        Detect grooming behavior.

        Grooming = compact body posture + low velocity.
        """
        # Must be relatively stationary
        if velocity > self.thresholds.grooming_velocity_max:
            return False

        # Check for compact body
        is_compact = pose_features.get('is_compact', False)
        body_length = pose_features.get('body_length', np.nan)

        if is_compact:
            return True

        # Alternative: check body length against reference
        if self.reference_body_length is not None and not np.isnan(body_length):
            ratio = body_length / self.reference_body_length
            if ratio < self.thresholds.grooming_body_compact_ratio:
                return True

        return False

    def _detect_wall_sniffing(self, pose_data: Dict, pose_features: Dict) -> bool:
        """
        Detect wall-sniffing behavior.

        Wall-sniffing = nose within threshold distance from any wall.
        """
        # Use nose position if available, otherwise centroid
        nose_x = pose_features.get('nose_x', np.nan)
        nose_y = pose_features.get('nose_y', np.nan)

        if np.isnan(nose_x) or np.isnan(nose_y):
            centroid = pose_data.get('centroid', (np.nan, np.nan))
            nose_x, nose_y = centroid

        if np.isnan(nose_x) or np.isnan(nose_y):
            return False

        # Calculate distance to each wall (in cm)
        wall_dist_cm = self.thresholds.wall_distance_max
        wall_dist_px = wall_dist_cm * self.pixels_per_cm

        # Check all four walls
        dist_to_left = nose_x
        dist_to_right = self.arena_width - nose_x
        dist_to_top = nose_y
        dist_to_bottom = self.arena_height - nose_y

        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

        return min_dist < wall_dist_px

    def _detect_center_exploration(self, pose_data: Dict, velocity: float) -> bool:
        """
        Detect center zone exploration.

        Center exploration = in center zone + actively moving.
        """
        centroid = pose_data.get('centroid', (np.nan, np.nan))

        if np.isnan(centroid[0]) or np.isnan(centroid[1]):
            return False

        # Must be moving (not freezing)
        if velocity < self.thresholds.freezing_velocity_max:
            return False

        # Check if in center zone
        center_pct = self.thresholds.center_zone_pct
        margin_x = self.arena_width * (1 - center_pct) / 2
        margin_y = self.arena_height * (1 - center_pct) / 2

        in_center = (
            margin_x <= centroid[0] <= self.arena_width - margin_x and
            margin_y <= centroid[1] <= self.arena_height - margin_y
        )

        return in_center

    def update_reference_body_length(self, body_length: float):
        """
        Update reference body length for grooming detection.

        Should be called with body length from frames where animal is
        clearly in extended posture (e.g., during locomotion).
        """
        if np.isnan(body_length):
            return

        if self.reference_body_length is None:
            self.reference_body_length = body_length
        else:
            # Running average with decay
            self.reference_body_length = 0.9 * self.reference_body_length + 0.1 * body_length

    def reset(self):
        """Reset classifier state for new video."""
        self.state_tracker.reset()
        self.reference_body_length = None


def consolidate_behaviors(
    frame_behaviors: List[List[str]],
    fps: float
) -> pd.DataFrame:
    """
    Convert frame-level behaviors into a structured DataFrame.

    Args:
        frame_behaviors: List of behavior lists, one per frame.
        fps: Video frame rate.

    Returns:
        DataFrame with columns:
            - frame: Frame number
            - time_s: Time in seconds
            - locomotion, rearing, grooming, freezing, wall_sniffing,
              center_exploration: Boolean columns for each behavior
    """
    all_behaviors = [
        'locomotion', 'rearing', 'grooming', 'freezing',
        'wall_sniffing', 'center_exploration'
    ]

    data = {
        'frame': list(range(len(frame_behaviors))),
        'time_s': [i / fps for i in range(len(frame_behaviors))]
    }

    # Initialize behavior columns
    for behavior in all_behaviors:
        data[behavior] = [False] * len(frame_behaviors)

    # Fill in behavior occurrences
    for i, behaviors in enumerate(frame_behaviors):
        for behavior in behaviors:
            if behavior in all_behaviors:
                data[behavior][i] = True

    return pd.DataFrame(data)


def extract_behavior_episodes(
    behavior_df: pd.DataFrame,
    fps: float
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Extract continuous episodes for each behavior.

    Args:
        behavior_df: DataFrame from consolidate_behaviors().
        fps: Video frame rate.

    Returns:
        Dictionary mapping behavior name to list of episodes.
        Each episode is (start_frame, end_frame, duration_s).
    """
    all_behaviors = [
        'locomotion', 'rearing', 'grooming', 'freezing',
        'wall_sniffing', 'center_exploration'
    ]

    episodes = {}

    for behavior in all_behaviors:
        if behavior not in behavior_df.columns:
            episodes[behavior] = []
            continue

        behavior_episodes = []
        signal = behavior_df[behavior].values

        # Find episode boundaries
        in_episode = False
        start_frame = 0

        for i, active in enumerate(signal):
            if active and not in_episode:
                # Episode start
                in_episode = True
                start_frame = i
            elif not active and in_episode:
                # Episode end
                in_episode = False
                end_frame = i - 1
                duration = (end_frame - start_frame + 1) / fps
                behavior_episodes.append((start_frame, end_frame, duration))

        # Handle episode that extends to end
        if in_episode:
            end_frame = len(signal) - 1
            duration = (end_frame - start_frame + 1) / fps
            behavior_episodes.append((start_frame, end_frame, duration))

        episodes[behavior] = behavior_episodes

    return episodes


def generate_behavior_summary(
    behavior_df: pd.DataFrame,
    episodes: Dict[str, List[Tuple[int, int, float]]],
    fps: float
) -> Dict:
    """
    Generate summary statistics for all behaviors.

    Args:
        behavior_df: DataFrame from consolidate_behaviors().
        episodes: Dictionary from extract_behavior_episodes().
        fps: Video frame rate.

    Returns:
        Dictionary with summary statistics for each behavior.
    """
    total_frames = len(behavior_df)
    total_time = total_frames / fps

    summary = {
        'total_time_s': total_time,
        'total_frames': total_frames
    }

    all_behaviors = [
        'locomotion', 'rearing', 'grooming', 'freezing',
        'wall_sniffing', 'center_exploration'
    ]

    for behavior in all_behaviors:
        if behavior in behavior_df.columns:
            active_frames = behavior_df[behavior].sum()
            active_time = active_frames / fps
            percentage = (active_frames / total_frames) * 100 if total_frames > 0 else 0

            behavior_episodes = episodes.get(behavior, [])
            num_episodes = len(behavior_episodes)

            avg_duration = 0
            if num_episodes > 0:
                avg_duration = sum(ep[2] for ep in behavior_episodes) / num_episodes

            summary[f'{behavior}_time_s'] = active_time
            summary[f'{behavior}_percentage'] = percentage
            summary[f'{behavior}_episodes'] = num_episodes
            summary[f'{behavior}_avg_duration_s'] = avg_duration
        else:
            summary[f'{behavior}_time_s'] = 0
            summary[f'{behavior}_percentage'] = 0
            summary[f'{behavior}_episodes'] = 0
            summary[f'{behavior}_avg_duration_s'] = 0

    return summary
