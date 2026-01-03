"""
metrics.py - Quantitative Behavioral Analysis Module for BioTrack-Lite

This module provides functions to calculate ethologically relevant metrics
from tracked position data. All calculations use vectorized NumPy operations
for efficiency on large datasets.

Key Metrics:
- Velocity: Instantaneous speed (cm/s)
- Total Distance: Cumulative path length (cm)
- Thigmotaxis: Wall-seeking behavior index (anxiety indicator)
- Freezing: Immobility detection (fear response indicator)
- Zone Crossings: Exploratory behavior measure

Design Philosophy:
- NaN values are preserved, not interpolated (scientific honesty)
- All spatial measurements are converted to real units (cm) for publication
- Functions are pure (no side effects) for testability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def calculate_velocity(
    df: pd.DataFrame,
    fps: float,
    pixels_per_cm: float
) -> pd.Series:
    """
    Calculate instantaneous velocity between consecutive frames.

    Velocity is computed as Euclidean distance traveled divided by time
    between frames, converted to cm/s.

    Args:
        df: DataFrame with 'x' and 'y' columns (pixel coordinates).
        fps: Video frame rate (frames per second).
        pixels_per_cm: Calibration factor for converting pixels to cm.

    Returns:
        pd.Series of velocity values in cm/s.
        First frame will be NaN (no previous position to compare).
        Frames following NaN positions will also be NaN.

    NaN Handling Philosophy:
        If either the current or previous position is NaN, velocity is NaN.
        We do NOT interpolate across missing frames because:
        1. The animal may have moved unpredictably during the gap
        2. Interpolation would mask tracking quality issues
        3. Scientific publications require honest data handling

    Example:
        >>> df['velocity_cm_s'] = calculate_velocity(df, fps=30, pixels_per_cm=10)
    """
    # Calculate displacement in pixels
    dx = df['x'].diff()
    dy = df['y'].diff()

    # Euclidean distance between consecutive frames
    distance_px = np.sqrt(dx**2 + dy**2)

    # Convert to cm
    distance_cm = distance_px / pixels_per_cm

    # Time between frames
    dt = 1.0 / fps

    # Velocity = distance / time
    velocity = distance_cm / dt

    return velocity


def calculate_total_distance(
    df: pd.DataFrame,
    pixels_per_cm: float
) -> float:
    """
    Calculate total distance traveled (cumulative path length).

    Only sums distances between valid (non-NaN) consecutive positions.
    This means gaps in tracking do not contribute to the total.

    Args:
        df: DataFrame with 'x' and 'y' columns (pixel coordinates).
        pixels_per_cm: Calibration factor for converting pixels to cm.

    Returns:
        Total distance traveled in cm.

    Why skip NaN segments:
        If tracking was lost for 10 frames, we don't know where the animal
        went during that time. Including an estimated distance would be
        speculation, not measurement.
    """
    # Calculate displacement in pixels
    dx = df['x'].diff()
    dy = df['y'].diff()

    # Euclidean distance between consecutive frames
    distance_px = np.sqrt(dx**2 + dy**2)

    # Sum valid distances only (NaN values are ignored by nansum)
    total_px = np.nansum(distance_px)

    # Convert to cm
    total_cm = total_px / pixels_per_cm

    return total_cm


def calculate_thigmotaxis(
    df: pd.DataFrame,
    arena_width: float,
    arena_height: float,
    center_zone_pct: float = 0.6
) -> Dict:
    """
    Calculate thigmotaxis index (wall-seeking behavior).

    Thigmotaxis is the tendency of rodents to stay close to walls,
    which is a well-established indicator of anxiety-like behavior.

    The arena is divided into:
    - Center Zone: Inner rectangle (default 60% of arena dimensions)
    - Wall Zone: Outer margin surrounding the center

    Args:
        df: DataFrame with 'x' and 'y' columns (pixel coordinates).
        arena_width: Width of tracking area in pixels.
        arena_height: Height of tracking area in pixels.
        center_zone_pct: Fraction of arena dimensions for center zone (0-1).
                        Default 0.6 means center is 60% x 60% of arena.

    Returns:
        Dictionary containing:
            - 'thigmotaxis_index': Fraction of time in wall zone (0-1)
            - 'center_time_frames': Number of frames in center
            - 'wall_time_frames': Number of frames in wall zone
            - 'center_zone_bounds': (x_min, x_max, y_min, y_max) of center

    Interpretation:
        - High thigmotaxis (>0.8): Strong wall preference (anxious)
        - Moderate (0.5-0.8): Normal exploratory behavior
        - Low (<0.5): Center preference (low anxiety, habituated)
    """
    # Calculate center zone boundaries
    margin_x = arena_width * (1 - center_zone_pct) / 2
    margin_y = arena_height * (1 - center_zone_pct) / 2

    center_x_min = margin_x
    center_x_max = arena_width - margin_x
    center_y_min = margin_y
    center_y_max = arena_height - margin_y

    # Determine which zone each frame is in
    # Valid positions only (exclude NaN)
    valid_mask = ~(df['x'].isna() | df['y'].isna())

    in_center = (
        (df['x'] >= center_x_min) &
        (df['x'] <= center_x_max) &
        (df['y'] >= center_y_min) &
        (df['y'] <= center_y_max) &
        valid_mask
    )

    in_wall = valid_mask & ~in_center

    center_frames = in_center.sum()
    wall_frames = in_wall.sum()
    total_valid = valid_mask.sum()

    # Calculate thigmotaxis index
    if total_valid > 0:
        thigmotaxis_index = wall_frames / total_valid
    else:
        thigmotaxis_index = np.nan

    return {
        'thigmotaxis_index': thigmotaxis_index,
        'center_time_frames': int(center_frames),
        'wall_time_frames': int(wall_frames),
        'total_valid_frames': int(total_valid),
        'center_zone_bounds': (center_x_min, center_x_max, center_y_min, center_y_max)
    }


def detect_freezing(
    df: pd.DataFrame,
    fps: float,
    velocity_threshold: float = 2.0,
    min_duration: float = 1.0
) -> Dict:
    """
    Detect freezing episodes (immobility periods).

    Freezing is defined as periods where velocity stays below a threshold
    for a minimum duration. This is a key behavioral indicator of fear
    or anxiety in rodents.

    Args:
        df: DataFrame with 'velocity_cm_s' column.
        fps: Video frame rate.
        velocity_threshold: Maximum velocity (cm/s) to be considered immobile.
                           Default 2.0 cm/s accounts for minor body movements.
        min_duration: Minimum duration (seconds) for a freezing episode.
                     Default 1.0 second is a common threshold in the literature.

    Returns:
        Dictionary containing:
            - 'freezing_episodes': List of (start_frame, end_frame, duration_s)
            - 'total_freezing_time_s': Total time spent freezing
            - 'freezing_percentage': Percentage of recording spent freezing
            - 'num_episodes': Number of distinct freezing bouts

    Scientific Context:
        Freezing is an innate fear response where the animal becomes
        completely still. It's commonly measured in fear conditioning
        experiments and elevated plus maze tests.
    """
    if 'velocity_cm_s' not in df.columns:
        raise ValueError("DataFrame must contain 'velocity_cm_s' column. "
                        "Run calculate_velocity() first.")

    min_frames = int(min_duration * fps)

    # Identify frames where animal is immobile
    # NaN velocities are not counted as immobile
    is_immobile = (df['velocity_cm_s'] < velocity_threshold) & ~df['velocity_cm_s'].isna()

    # Find continuous immobile segments
    episodes = []
    in_episode = False
    episode_start = 0

    for i, immobile in enumerate(is_immobile):
        if immobile and not in_episode:
            # Start of potential episode
            in_episode = True
            episode_start = i
        elif not immobile and in_episode:
            # End of episode
            episode_end = i
            episode_length = episode_end - episode_start

            if episode_length >= min_frames:
                duration_s = episode_length / fps
                episodes.append((episode_start, episode_end, duration_s))

            in_episode = False

    # Handle episode that extends to end of recording
    if in_episode:
        episode_end = len(is_immobile)
        episode_length = episode_end - episode_start

        if episode_length >= min_frames:
            duration_s = episode_length / fps
            episodes.append((episode_start, episode_end, duration_s))

    # Calculate summary statistics
    total_freezing_time = sum(ep[2] for ep in episodes)
    total_recording_time = len(df) / fps

    if total_recording_time > 0:
        freezing_pct = (total_freezing_time / total_recording_time) * 100
    else:
        freezing_pct = 0.0

    return {
        'freezing_episodes': episodes,
        'total_freezing_time_s': total_freezing_time,
        'freezing_percentage': freezing_pct,
        'num_episodes': len(episodes)
    }


def count_zone_crossings(
    df: pd.DataFrame,
    arena_width: float,
    arena_height: float,
    center_zone_pct: float = 0.6
) -> int:
    """
    Count the number of transitions between center and wall zones.

    Zone crossings indicate exploratory behavior. Animals that are
    actively exploring will cross zone boundaries frequently, while
    anxious animals tend to stay in one zone (usually the wall zone).

    Args:
        df: DataFrame with 'x' and 'y' columns (pixel coordinates).
        arena_width: Width of tracking area in pixels.
        arena_height: Height of tracking area in pixels.
        center_zone_pct: Fraction of arena dimensions for center zone (0-1).

    Returns:
        Number of zone transitions (both center→wall and wall→center).

    Interpretation:
        - High crossings: Active exploration (curious, low anxiety)
        - Low crossings: Avoidant behavior (anxious, or habituated to zone)
    """
    # Calculate center zone boundaries (same as thigmotaxis)
    margin_x = arena_width * (1 - center_zone_pct) / 2
    margin_y = arena_height * (1 - center_zone_pct) / 2

    center_x_min = margin_x
    center_x_max = arena_width - margin_x
    center_y_min = margin_y
    center_y_max = arena_height - margin_y

    # Determine zone for each frame
    # 1 = center, 0 = wall, NaN = unknown
    in_center = (
        (df['x'] >= center_x_min) &
        (df['x'] <= center_x_max) &
        (df['y'] >= center_y_min) &
        (df['y'] <= center_y_max)
    ).astype(float)

    # Mark NaN positions as NaN zone
    invalid = df['x'].isna() | df['y'].isna()
    in_center[invalid] = np.nan

    # Count transitions (changes in zone)
    # diff() will be non-zero when zone changes
    zone_changes = in_center.diff().abs()

    # Only count valid transitions (not involving NaN)
    crossings = int(np.nansum(zone_changes))

    return crossings


def generate_summary_stats(
    df: pd.DataFrame,
    fps: float,
    pixels_per_cm: float,
    arena_width_px: float,
    arena_height_px: float,
    center_zone_pct: float = 0.6,
    freezing_velocity_threshold: float = 2.0,
    freezing_min_duration: float = 1.0
) -> Dict:
    """
    Generate comprehensive summary statistics for the tracking session.

    This is a convenience function that calculates all metrics and
    returns them in a publication-ready format.

    Args:
        df: DataFrame with 'x', 'y', and optionally 'velocity_cm_s' columns.
        fps: Video frame rate.
        pixels_per_cm: Calibration factor.
        arena_width_px: Arena width in pixels.
        arena_height_px: Arena height in pixels.
        center_zone_pct: Center zone size fraction.
        freezing_velocity_threshold: Velocity threshold for freezing detection.
        freezing_min_duration: Minimum duration for freezing episodes.

    Returns:
        Dictionary with all summary metrics suitable for export.
    """
    # Ensure velocity is calculated
    if 'velocity_cm_s' not in df.columns:
        df = df.copy()
        df['velocity_cm_s'] = calculate_velocity(df, fps, pixels_per_cm)

    # Calculate all metrics
    total_distance = calculate_total_distance(df, pixels_per_cm)

    thigmotaxis = calculate_thigmotaxis(
        df, arena_width_px, arena_height_px, center_zone_pct
    )

    freezing = detect_freezing(
        df, fps, freezing_velocity_threshold, freezing_min_duration
    )

    zone_crossings = count_zone_crossings(
        df, arena_width_px, arena_height_px, center_zone_pct
    )

    # Velocity statistics (excluding NaN)
    valid_velocities = df['velocity_cm_s'].dropna()

    if len(valid_velocities) > 0:
        avg_velocity = valid_velocities.mean()
        max_velocity = valid_velocities.max()
        velocity_std = valid_velocities.std()
    else:
        avg_velocity = np.nan
        max_velocity = np.nan
        velocity_std = np.nan

    # Tracking quality metrics
    total_frames = len(df)
    valid_frames = (~df['x'].isna()).sum()
    tracking_quality = (valid_frames / total_frames * 100) if total_frames > 0 else 0

    # Recording duration
    duration_s = total_frames / fps

    return {
        # Session info
        'total_frames': total_frames,
        'duration_s': duration_s,
        'fps': fps,
        'tracking_quality_pct': tracking_quality,

        # Locomotion
        'total_distance_cm': total_distance,
        'avg_velocity_cm_s': avg_velocity,
        'max_velocity_cm_s': max_velocity,
        'velocity_std_cm_s': velocity_std,

        # Thigmotaxis
        'thigmotaxis_index': thigmotaxis['thigmotaxis_index'],
        'center_time_s': thigmotaxis['center_time_frames'] / fps,
        'wall_time_s': thigmotaxis['wall_time_frames'] / fps,

        # Freezing
        'freezing_time_s': freezing['total_freezing_time_s'],
        'freezing_percentage': freezing['freezing_percentage'],
        'freezing_episodes': freezing['num_episodes'],

        # Exploration
        'zone_crossings': zone_crossings,

        # Calibration (for reference)
        'pixels_per_cm': pixels_per_cm,
        'arena_width_px': arena_width_px,
        'arena_height_px': arena_height_px
    }


def prepare_export_dataframe(
    df: pd.DataFrame,
    fps: float,
    pixels_per_cm: float
) -> pd.DataFrame:
    """
    Prepare the tracking data for CSV export with all computed columns.

    Args:
        df: Raw DataFrame with 'frame', 'x', 'y' columns.
        fps: Video frame rate.
        pixels_per_cm: Calibration factor.

    Returns:
        DataFrame with columns:
            - frame: Frame number
            - time_s: Time in seconds
            - x_px, y_px: Position in pixels
            - x_cm, y_cm: Position in cm (from top-left origin)
            - velocity_cm_s: Instantaneous velocity
    """
    export_df = df.copy()

    # Time column
    export_df['time_s'] = export_df['frame'] / fps

    # Rename pixel columns
    export_df['x_px'] = export_df['x']
    export_df['y_px'] = export_df['y']

    # Convert to cm
    export_df['x_cm'] = export_df['x'] / pixels_per_cm
    export_df['y_cm'] = export_df['y'] / pixels_per_cm

    # Calculate velocity if not present
    if 'velocity_cm_s' not in export_df.columns:
        export_df['velocity_cm_s'] = calculate_velocity(export_df, fps, pixels_per_cm)

    # Select and order columns for export
    columns = ['frame', 'time_s', 'x_px', 'y_px', 'x_cm', 'y_cm', 'velocity_cm_s']

    return export_df[columns]
