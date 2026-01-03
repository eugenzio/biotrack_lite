"""
visualization.py - Plotting Module for BioTrack-Lite

This module provides publication-quality visualizations for behavioral data.
All plots are designed to be interpretable and suitable for scientific
publications or presentations.

Design Principles:
- All functions return matplotlib Figure objects (not Axes)
- Figures are self-contained with labels and legends
- Color schemes are colorblind-friendly where possible
- Image coordinate system (Y-axis inverted) is preserved for spatial plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from typing import Dict, List, Tuple, Optional


def plot_trajectory(
    df: pd.DataFrame,
    arena_dims: Optional[Tuple[float, float]] = None,
    center_zone_pct: float = 0.6,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot the animal's trajectory colored by velocity.

    Creates a scatter plot where each point represents the animal's position
    at a given frame, with color intensity indicating velocity.

    Args:
        df: DataFrame with 'x', 'y', and 'velocity_cm_s' columns.
        arena_dims: (width, height) in pixels. If provided, draws arena boundary.
        center_zone_pct: Center zone fraction for boundary visualization.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.

    Visual Design:
        - Viridis colormap: perceptually uniform, colorblind-friendly
        - Y-axis inverted to match image coordinates (0,0 at top-left)
        - Center zone shown as dashed rectangle for reference
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter valid data points
    valid = ~(df['x'].isna() | df['y'].isna())
    x = df.loc[valid, 'x']
    y = df.loc[valid, 'y']

    # Get velocity for coloring (use 0 if not available)
    if 'velocity_cm_s' in df.columns:
        velocity = df.loc[valid, 'velocity_cm_s'].fillna(0)
    else:
        velocity = np.zeros(len(x))

    # Create scatter plot
    scatter = ax.scatter(
        x, y,
        c=velocity,
        cmap='viridis',
        s=10,
        alpha=0.6,
        edgecolors='none'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Velocity (cm/s)')

    # Draw arena boundary if dimensions provided
    if arena_dims is not None:
        width, height = arena_dims

        # Arena boundary
        arena_rect = patches.Rectangle(
            (0, 0), width, height,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(arena_rect)

        # Center zone boundary
        margin_x = width * (1 - center_zone_pct) / 2
        margin_y = height * (1 - center_zone_pct) / 2
        center_rect = patches.Rectangle(
            (margin_x, margin_y),
            width * center_zone_pct,
            height * center_zone_pct,
            linewidth=1.5, edgecolor='gray', facecolor='none',
            linestyle='--', label='Center Zone'
        )
        ax.add_patch(center_rect)
        ax.legend(loc='upper right')

        ax.set_xlim(-10, width + 10)
        ax.set_ylim(-10, height + 10)

    # Invert Y-axis to match image coordinates
    ax.invert_yaxis()

    # Labels
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title('Movement Trajectory', fontsize=14)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    arena_dims: Tuple[float, float],
    center_zone_pct: float = 0.6,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot a spatial density heatmap of the animal's position.

    Uses kernel density estimation (KDE) to show where the animal
    spent most of its time during the recording.

    Args:
        df: DataFrame with 'x' and 'y' columns.
        arena_dims: (width, height) in pixels.
        center_zone_pct: Center zone fraction for boundary visualization.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.

    Scientific Use:
        Heatmaps reveal spatial preferences such as:
        - Corner preference (high anxiety)
        - Center avoidance (normal thigmotaxis)
        - Uniform distribution (habituation/exploration)
    """
    fig, ax = plt.subplots(figsize=figsize)

    width, height = arena_dims

    # Filter valid data
    valid = ~(df['x'].isna() | df['y'].isna())
    x = df.loc[valid, 'x']
    y = df.loc[valid, 'y']

    if len(x) < 10:
        ax.text(
            0.5, 0.5, 'Insufficient data for heatmap',
            ha='center', va='center', transform=ax.transAxes, fontsize=14
        )
        return fig

    # Create KDE plot
    try:
        sns.kdeplot(
            x=x, y=y,
            cmap='YlOrRd',
            fill=True,
            levels=20,
            alpha=0.7,
            ax=ax
        )
    except Exception:
        # Fallback to 2D histogram if KDE fails
        ax.hist2d(x, y, bins=30, cmap='YlOrRd', alpha=0.7)

    # Draw arena boundary
    arena_rect = patches.Rectangle(
        (0, 0), width, height,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(arena_rect)

    # Draw center zone
    margin_x = width * (1 - center_zone_pct) / 2
    margin_y = height * (1 - center_zone_pct) / 2
    center_rect = patches.Rectangle(
        (margin_x, margin_y),
        width * center_zone_pct,
        height * center_zone_pct,
        linewidth=1.5, edgecolor='white', facecolor='none',
        linestyle='--', label='Center Zone'
    )
    ax.add_patch(center_rect)

    # Invert Y-axis to match image coordinates
    ax.invert_yaxis()

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Inverted
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title('Spatial Density Heatmap', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_velocity_timeseries(
    df: pd.DataFrame,
    fps: float,
    freezing_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot velocity over time with optional freezing threshold.

    Shows the animal's activity level throughout the recording.

    Args:
        df: DataFrame with 'frame' and 'velocity_cm_s' columns.
        fps: Video frame rate (for time axis conversion).
        freezing_threshold: If provided, draws horizontal line at threshold
                           and highlights periods below it.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.

    Scientific Use:
        - Identify activity patterns over time
        - Detect habituation (decreasing activity)
        - Visualize freezing bouts
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert frame to time
    time = df['frame'] / fps

    # Plot velocity
    ax.plot(
        time,
        df['velocity_cm_s'],
        color='steelblue',
        linewidth=0.8,
        alpha=0.8,
        label='Velocity'
    )

    # Highlight freezing threshold if provided
    if freezing_threshold is not None:
        ax.axhline(
            y=freezing_threshold,
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label=f'Freezing threshold ({freezing_threshold} cm/s)'
        )

        # Shade regions below threshold
        below_threshold = df['velocity_cm_s'] < freezing_threshold
        ax.fill_between(
            time,
            0,
            df['velocity_cm_s'],
            where=below_threshold,
            color='red',
            alpha=0.2
        )

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Velocity (cm/s)', fontsize=12)
    ax.set_title('Velocity Over Time', fontsize=14)
    ax.set_xlim(0, time.max())
    ax.set_ylim(0, None)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_zone_occupancy(
    thigmotaxis_data: Dict,
    fps: float,
    figsize: Tuple[int, int] = (6, 6)
) -> plt.Figure:
    """
    Plot zone occupancy as a pie chart.

    Shows the proportion of time spent in center vs wall zones.

    Args:
        thigmotaxis_data: Dictionary from calculate_thigmotaxis().
        fps: Video frame rate (for time conversion).
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    center_time = thigmotaxis_data['center_time_frames'] / fps
    wall_time = thigmotaxis_data['wall_time_frames'] / fps

    # Data for pie chart
    sizes = [center_time, wall_time]
    labels = [
        f'Center\n{center_time:.1f}s ({100 - thigmotaxis_data["thigmotaxis_index"]*100:.1f}%)',
        f'Wall Zone\n{wall_time:.1f}s ({thigmotaxis_data["thigmotaxis_index"]*100:.1f}%)'
    ]
    colors = ['#2ecc71', '#e74c3c']  # Green for center, red for wall
    explode = (0.05, 0)

    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )

    ax.set_title('Zone Occupancy', fontsize=14)

    # Add thigmotaxis index annotation
    ax.annotate(
        f'Thigmotaxis Index: {thigmotaxis_data["thigmotaxis_index"]:.3f}',
        xy=(0, -1.3),
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()
    return fig


def plot_freezing_episodes(
    df: pd.DataFrame,
    freezing_data: Dict,
    fps: float,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot freezing episodes as a timeline.

    Shows when and for how long the animal was immobile.

    Args:
        df: DataFrame with 'frame' and 'velocity_cm_s' columns.
        freezing_data: Dictionary from detect_freezing().
        fps: Video frame rate.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.

    Scientific Use:
        Visualize the temporal distribution of freezing bouts.
        Clustered freezing may indicate specific fear triggers.
    """
    fig, ax = plt.subplots(figsize=figsize)

    total_time = len(df) / fps

    # Create timeline
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, 1)

    # Draw freezing episodes as red bars
    for episode in freezing_data['freezing_episodes']:
        start_frame, end_frame, duration = episode
        start_time = start_frame / fps
        end_time = end_frame / fps

        ax.barh(
            y=0.5,
            width=end_time - start_time,
            left=start_time,
            height=0.6,
            color='#e74c3c',
            alpha=0.8,
            edgecolor='darkred',
            linewidth=1
        )

    # Draw non-freezing periods as green background
    ax.axhspan(0.2, 0.8, facecolor='#2ecc71', alpha=0.2, zorder=0)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title(
        f'Freezing Episodes (n={freezing_data["num_episodes"]}, '
        f'total={freezing_data["total_freezing_time_s"]:.1f}s, '
        f'{freezing_data["freezing_percentage"]:.1f}%)',
        fontsize=14
    )
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label='Freezing'),
        Patch(facecolor='#2ecc71', alpha=0.2, label='Active')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


def plot_summary_dashboard(
    df: pd.DataFrame,
    summary_stats: Dict,
    thigmotaxis_data: Dict,
    freezing_data: Dict,
    arena_dims: Tuple[float, float],
    fps: float,
    center_zone_pct: float = 0.6,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive summary dashboard with multiple plots.

    Combines trajectory, heatmap, velocity, zone occupancy, and freezing
    into a single publication-ready figure.

    Args:
        df: DataFrame with tracking data.
        summary_stats: Dictionary from generate_summary_stats().
        thigmotaxis_data: Dictionary from calculate_thigmotaxis().
        freezing_data: Dictionary from detect_freezing().
        arena_dims: (width, height) in pixels.
        fps: Video frame rate.
        center_zone_pct: Center zone fraction.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)

    # Create grid: 3 rows, 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Top left: Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    valid = ~(df['x'].isna() | df['y'].isna())
    x = df.loc[valid, 'x']
    y = df.loc[valid, 'y']
    velocity = df.loc[valid, 'velocity_cm_s'].fillna(0) if 'velocity_cm_s' in df.columns else np.zeros(len(x))

    scatter = ax1.scatter(x, y, c=velocity, cmap='viridis', s=5, alpha=0.6)
    ax1.invert_yaxis()
    ax1.set_title('Trajectory')
    ax1.set_aspect('equal')
    plt.colorbar(scatter, ax=ax1, label='Velocity (cm/s)')

    # Top center: Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    if len(x) >= 10:
        try:
            sns.kdeplot(x=x, y=y, cmap='YlOrRd', fill=True, levels=15, ax=ax2)
        except Exception:
            ax2.hist2d(x, y, bins=20, cmap='YlOrRd')
    ax2.invert_yaxis()
    ax2.set_title('Spatial Density')
    ax2.set_aspect('equal')

    # Top right: Zone occupancy (pie)
    ax3 = fig.add_subplot(gs[0, 2])
    center_time = thigmotaxis_data['center_time_frames'] / fps
    wall_time = thigmotaxis_data['wall_time_frames'] / fps
    if center_time + wall_time > 0:
        ax3.pie(
            [center_time, wall_time],
            labels=['Center', 'Wall'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90
        )
    ax3.set_title('Zone Occupancy')

    # Middle row: Velocity time series (full width)
    ax4 = fig.add_subplot(gs[1, :])
    time = df['frame'] / fps
    ax4.plot(time, df['velocity_cm_s'], color='steelblue', linewidth=0.5, alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (cm/s)')
    ax4.set_title('Velocity Over Time')
    ax4.set_xlim(0, time.max())
    ax4.grid(True, alpha=0.3)

    # Bottom left: Freezing timeline
    ax5 = fig.add_subplot(gs[2, :2])
    total_time = len(df) / fps
    ax5.set_xlim(0, total_time)
    ax5.set_ylim(0, 1)
    ax5.axhspan(0.2, 0.8, facecolor='#2ecc71', alpha=0.2)
    for episode in freezing_data['freezing_episodes']:
        start_frame, end_frame, duration = episode
        ax5.barh(0.5, (end_frame - start_frame) / fps, left=start_frame / fps,
                height=0.6, color='#e74c3c', alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_title(f'Freezing Episodes (n={freezing_data["num_episodes"]})')
    ax5.set_yticks([])

    # Bottom right: Summary stats table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    stats_text = (
        f"Session Summary\n"
        f"{'='*30}\n"
        f"Duration: {summary_stats['duration_s']:.1f} s\n"
        f"Tracking Quality: {summary_stats['tracking_quality_pct']:.1f}%\n\n"
        f"Locomotion\n"
        f"{'='*30}\n"
        f"Total Distance: {summary_stats['total_distance_cm']:.1f} cm\n"
        f"Avg Velocity: {summary_stats['avg_velocity_cm_s']:.2f} cm/s\n"
        f"Max Velocity: {summary_stats['max_velocity_cm_s']:.2f} cm/s\n\n"
        f"Behavior\n"
        f"{'='*30}\n"
        f"Thigmotaxis: {summary_stats['thigmotaxis_index']:.3f}\n"
        f"Freezing: {summary_stats['freezing_percentage']:.1f}%\n"
        f"Zone Crossings: {summary_stats['zone_crossings']}"
    )

    ax6.text(
        0.1, 0.95, stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.suptitle('BioTrack-Lite Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig
