"""
app.py - Professional Research Dashboard for Movyent

This is the main entry point for the Movyent application.
It provides a web-based interface for video analysis, parameter tuning,
and result export.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
from pathlib import Path
from io import BytesIO
import zipfile

# Import local modules
from tracker import RodentTracker, BodyParts
from metrics import (
    calculate_velocity,
    calculate_total_distance,
    calculate_thigmotaxis,
    detect_freezing,
    count_zone_crossings,
    generate_summary_stats,
    prepare_export_dataframe
)
from visualization import (
    plot_trajectory,
    plot_heatmap,
    plot_velocity_timeseries,
    plot_zone_occupancy,
    plot_freezing_episodes,
    plot_summary_dashboard
)


# Professional Research Theme CSS
PROFESSIONAL_THEME = """
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* CSS Variables for consistent theming */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #e9ecef;
        --bg-hover: #dee2e6;
        --border-primary: #dee2e6;
        --border-secondary: #e9ecef;
        --text-primary: #000000;
        --text-secondary: #4a5568;
        --text-muted: #718096;
        --accent-primary: #2563eb;
        --accent-secondary: #1d4ed8;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes buttonClick {
        0% { transform: scale(1); }
        50% { transform: scale(0.95); }
        100% { transform: scale(1); }
    }

    /* Main app background */
    .stApp {
        background: var(--bg-primary);
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        animation: fadeIn 0.8s ease-out;
    }

    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Headers */
    h1 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 2rem !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }

    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    /* Text */
    .stMarkdown, p, span, label {
        color: var(--text-secondary);
    }

    /* Metric cards - refined */
    [data-testid="stMetric"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1.25rem;
    }

    [data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Primary button - clean blue */
    .stButton > button, 
    .stButton > button[kind="primary"],
    div[data-testid="stButton"] > button {
        background: var(--accent-primary) !important;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 0.9rem;
    }

    /* Ensure text inside button is white */
    .stButton > button p,
    .stButton > button[kind="primary"] p {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background: var(--accent-secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
    }

    .stButton > button:active {
        animation: buttonClick 0.2s ease-in-out;
        transform: translateY(0);
    }

    /* Download buttons - subtle */
    .stDownloadButton > button {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        color: var(--text-primary) !important;
        border-radius: 8px;
        font-weight: 500;
    }

    .stDownloadButton > button:hover {
        background: var(--bg-hover);
        border-color: var(--accent-primary);
    }

    /* Tabs - minimal */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid var(--border-primary);
        gap: 0;
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 0;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--accent-primary) !important;
        border-bottom: 2px solid var(--accent-primary) !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: var(--bg-secondary);
        border-radius: 0 0 12px 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-primary);
        border-top: none;
    }

    /* File uploader - WHITE BOX, BLACK TEXT */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2px dashed #ccc !important;
        border-radius: 12px;
        padding: 2rem;
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] .stMarkdown,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small {
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] button {
        color: #000000 !important;
        border-color: #ccc !important;
        background: #f8f9fa !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-primary) !important;
    }

    /* Sliders - refined */
    .stSlider > div > div > div {
        background: var(--border-primary) !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent-primary) !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 4px;
    }

    /* DataFrame - clean table */
    [data-testid="stDataFrame"] {
        background: var(--bg-secondary);
        border-radius: 12px;
        border: 1px solid var(--border-primary);
        overflow: hidden;
    }

    /* Expander - minimal */
    [data-testid="stExpander"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        padding: 1rem 1.25rem;
        font-weight: 500;
        color: var(--text-primary);
    }

    [data-testid="stExpander"] summary:hover {
        background: var(--bg-tertiary);
    }

    /* Number input */
    .stNumberInput input {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        color: var(--text-primary);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
    }

    .stNumberInput input:focus {
        border-color: var(--accent-primary);
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        border-radius: 8px;
    }

    /* Checkbox */
    .stCheckbox label span {
        color: var(--text-secondary) !important;
    }

    /* Image containers - WHITE BOX */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #ccc;
        background: #ffffff !important;
        padding: 10px;
    }
    
    [data-testid="stImage"] caption, 
    [data-testid="stImage"] .stCaption {
        color: #000000 !important;
        font-weight: 500;
    }

    /* Caption (Global fallback) */
    .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.8rem;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: 8px;
        border-left: 3px solid var(--accent-primary);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-primary);
        border-radius: 5px;
        border: 2px solid var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* MEDIA PLAYER SPECIAL STYLING */
    .media-player-container {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .media-player-container .time-display {
        color: #000000 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        padding-top: 8px;
    }
    
    .media-player-container .stButton > button {
        background: #f8f9fa !important;
        color: #000000 !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.5rem 0.75rem;
        min-width: 60px;
        height: 36px;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .media-player-container .stButton > button:hover {
        background: #e9ecef !important;
        border-color: #ced4da !important;
        transform: translateY(-1px);
    }
    
    .media-player-container .stButton > button:active {
        background: #dde2e6 !important;
        transform: translateY(0);
    }

    /* Highlight play/pause button override */
    .media-player-container [data-testid="stHorizontalBlock"] > div:nth-child(4) button {
        background: #238636 !important;
        border-color: #238636 !important;
        color: #ffffff !important;
        min-width: 70px;
    }
    
    .media-player-container [data-testid="stHorizontalBlock"] > div:nth-child(4) button p {
        color: #ffffff !important;
    }

    .media-player-container [data-testid="stHorizontalBlock"] > div:nth-child(4) button:hover {
        background: #2ea043 !important;
        border-color: #2ea043 !important;
    }
    
    .media-player-container .stSelectbox > div > div {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        color: #000000 !important;
    }
    
    .media-player-container .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
</style>
"""


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="Movyent | Behavioral Analysis",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'tracking_data' not in st.session_state:
        st.session_state.tracking_data = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = None
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0.0

    # Apply professional theme
    st.markdown(PROFESSIONAL_THEME, unsafe_allow_html=True)

    # Title (centered)
    st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
<h1 style="margin-bottom: 0.5rem;">Movyent</h1>
<p style="color: #4a5568; font-size: 0.95rem; margin-bottom: 1.5rem;">
Behavioral Analysis System | MOG2 Background Subtraction
</p>
<div style="display: inline-flex; align-items: center; gap: 0.75rem; background: rgba(233, 236, 239, 0.8); border: 1px solid #dee2e6; padding: 0.75rem 1.5rem; border-radius: 50px;">
<span style="color: #000000; font-size: 0.95rem; font-weight: 500; letter-spacing: 0.01em;">Designed for low-spec computers and Chromebooks</span>
<img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/Google_Chrome_icon_%28February_2022%29.svg" width="24" alt="Chrome">
</div>
</div>
""", unsafe_allow_html=True)

    # Check if we have completed analysis results to show
    if st.session_state.analysis_complete and st.session_state.tracking_data is not None:
        show_analysis_results("")
        return

    # Center the file uploader
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV",
            label_visibility="collapsed"
        )

    if uploaded_file is None:
        show_welcome_screen()
        return

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Open video and get properties
    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():
        st.error("Could not open video file. Please try a different format.")
        cleanup_temp(temp_dir)
        return

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / detected_fps if detected_fps > 0 else 0

    # Video info display (centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background: #ffffff; padding: 1rem; border-radius: 12px; margin: 1rem 0; text-align: center; border: 1px solid #dee2e6;">
            <div style="color: #38a169; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                Video Loaded
            </div>
            <div style="color: #000000; font-family: 'IBM Plex Mono', monospace;">
                {total_frames:,} frames | {duration_s:.1f}s | {frame_width}x{frame_height}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Settings in expanders
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Calibration & ROI", expanded=False):
            # FPS
            use_detected_fps = st.checkbox(
                f"Use detected FPS ({detected_fps:.2f})",
                value=True if detected_fps > 0 else False
            )
            if use_detected_fps and detected_fps > 0:
                fps = detected_fps
            else:
                fps = st.number_input("Manual FPS", min_value=1.0, max_value=120.0, value=30.0, step=1.0)

            # Arena
            arena_width_cm = st.number_input("Arena Width (cm)", min_value=1.0, max_value=500.0, value=40.0, step=1.0)

            # ROI
            st.caption("ROI Cropping (%)")
            c1, c2 = st.columns(2)
            with c1:
                roi_top = st.slider("Top", 0, 30, 0, key="roi_top")
                roi_left = st.slider("Left", 0, 30, 0, key="roi_left")
            with c2:
                roi_bottom = st.slider("Bottom", 0, 30, 0, key="roi_bottom")
                roi_right = st.slider("Right", 0, 30, 0, key="roi_right")

    with col2:
        with st.expander("Tracking Parameters", expanded=False):
            var_threshold = st.slider("MOG2 Variance", 1, 100, 16, help="Lower = more sensitive")
            shadow_threshold = st.slider("Shadow Threshold", 200, 255, 250)
            min_contour_area = st.slider("Min Contour Area (px)", 100, 2000, 500)
            freezing_velocity = st.slider("Freezing Velocity (cm/s)", 0.5, 5.0, 2.0, 0.1)
            freezing_duration = st.slider("Freezing Duration (s)", 0.5, 3.0, 1.0, 0.1)
            center_zone_pct = st.slider("Center Zone (%)", 40, 80, 60) / 100.0
            preview_interval = st.slider("Preview Interval", 1, 30, 10)

            st.markdown("---")
            detect_body_parts = st.checkbox(
                "Detect Body Parts (Head/Body/Tail)",
                value=True,
                help="Estimate head, body center, and tail positions using contour geometry"
            )

    roi_pct = {'top': roi_top, 'bottom': roi_bottom, 'left': roi_left, 'right': roi_right}
    crop_width = int(frame_width * (100 - roi_left - roi_right) / 100)
    crop_height = int(frame_height * (100 - roi_top - roi_bottom) / 100)
    pixels_per_cm = crop_width / arena_width_cm

    # Preview and Start button
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_complete = False
            st.session_state.tracking_data = None
            st.session_state.is_playing = False
            st.session_state.current_time = 0.0

            run_analysis(
                video_path=temp_path,
                temp_dir=temp_dir,
                total_frames=total_frames,
                fps=fps,
                pixels_per_cm=pixels_per_cm,
                roi_pct=roi_pct,
                crop_width=crop_width,
                crop_height=crop_height,
                var_threshold=var_threshold,
                shadow_threshold=shadow_threshold,
                min_contour_area=min_contour_area,
                freezing_velocity=freezing_velocity,
                freezing_duration=freezing_duration,
                center_zone_pct=center_zone_pct,
                preview_interval=preview_interval,
                detect_body_parts=detect_body_parts
            )

    # Show preview
    show_preview(cap, roi_pct)
    cap.release()


def show_analysis_results(temp_dir: str):
    """Show analysis results with frame review slider."""

    # Get data from session state
    tracking_data = st.session_state.tracking_data
    video_path = st.session_state.video_path
    params = st.session_state.analysis_params

    fps = params['fps']
    pixels_per_cm = params['pixels_per_cm']
    roi_pct = params['roi_pct']
    crop_width = params['crop_width']
    crop_height = params['crop_height']
    var_threshold = params['var_threshold']
    shadow_threshold = params['shadow_threshold']
    min_contour_area = params['min_contour_area']
    freezing_velocity = params['freezing_velocity']
    freezing_duration = params['freezing_duration']
    center_zone_pct = params['center_zone_pct']
    total_frames = params['total_frames']
    detect_body_parts = params.get('detect_body_parts', False)

    # Calculate total duration
    total_duration = total_frames / fps

    # Helper function to format time as MM:SS
    def format_time(seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    # Callback for slider
    def update_from_slider():
        st.session_state.current_time = st.session_state.time_slider

    # --- Playback Logic (Pre-render) ---
    if st.session_state.is_playing:
        playback_speed = st.session_state.get('playback_speed_val', 1.0)
        time_step = (1.0 / fps) * playback_speed
        next_time = st.session_state.current_time + time_step

        if next_time >= total_duration:
            st.session_state.is_playing = False
            st.session_state.current_time = 0.0
        else:
            st.session_state.current_time = next_time

    # --- Modern Media Player UI ---
    st.markdown("""
    <style>
        .media-player-container {
            background: #ffffff;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
        }
        .media-player-container .stButton > button {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #dee2e6 !important;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            padding: 0.5rem 0.75rem;
            min-width: 60px;
            height: 36px;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .media-player-container .stButton > button:hover {
            background: #f8f9fa !important;
            border-color: #4a5568 !important;
        }
        .media-player-container .stButton > button:active {
            background: #e9ecef !important;
        }
        /* Highlight play/pause button (4th column in controls row) */
        .media-player-container [data-testid="stHorizontalBlock"] > div:nth-child(4) button {
            background: #38a169 !important;
            border-color: #38a169 !important;
            color: #ffffff !important;
            min-width: 70px;
        }
        .media-player-container [data-testid="stHorizontalBlock"] > div:nth-child(4) button:hover {
            background: #2f855a !important;
            border-color: #2f855a !important;
        }

        /* Align all control columns vertically */
        .media-player-container [data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .media-player-container div[data-testid="stVerticalBlock"] {
            gap: 0;
        }
        .media-player-container .stSelectbox > div > div {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
        }
        .media-player-container .stSelectbox label {
            display: none;
        }
        .time-display {
            color: #000000;
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            padding-top: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="media-player-container">', unsafe_allow_html=True)
        
        # --- Progress Bar ---
        slider_cols = st.columns([1, 10, 1])
        with slider_cols[0]:
            st.markdown(f'<div class="time-display" style="text-align: left;">{format_time(st.session_state.current_time)}</div>', unsafe_allow_html=True)

        with slider_cols[1]:
            st.slider(
                "Time",
                min_value=0.0,
                max_value=total_duration,
                value=st.session_state.current_time,
                key="time_slider",
                on_change=update_from_slider,
                format="",
                label_visibility="collapsed"
            )
        
        with slider_cols[2]:
            st.markdown(f'<div class="time-display" style="text-align: right;">{format_time(total_duration)}</div>', unsafe_allow_html=True)

        # --- Playback Controls ---
        # Use columns for centering and layout
        left_spacer, restart_col, rewind_col, play_col, forward_col, speed_col, frame_col, right_spacer = st.columns([2, 1, 1, 1, 1, 1, 2, 2])

        with restart_col:
            if st.button("Restart", key="restart_btn", icon=":material/skip_previous:", use_container_width=True, help="Return to start"):
                st.session_state.current_time = 0.0
                st.session_state.is_playing = False
                st.rerun()

        with rewind_col:
            if st.button("-5s", key="rewind_btn", icon=":material/fast_rewind:", use_container_width=True, help="Rewind 5 seconds"):
                st.session_state.current_time = max(0, st.session_state.current_time - 5)
                st.rerun()

        with play_col:
            play_label = "Pause" if st.session_state.is_playing else "Play"
            play_icon = ":material/pause:" if st.session_state.is_playing else ":material/play_arrow:"
            if st.button(play_label, key="play_pause_btn", icon=play_icon, use_container_width=True, help="Play/Pause"):
                st.session_state.is_playing = not st.session_state.is_playing
                st.rerun()

        with forward_col:
            if st.button("+5s", key="forward_btn", icon=":material/fast_forward:", use_container_width=True, help="Forward 5 seconds"):
                st.session_state.current_time = min(total_duration, st.session_state.current_time + 5)
                st.rerun()

        with speed_col:
            playback_speed = st.selectbox(
                "Speed",
                options=[0.25, 0.5, 1.0, 2.0, 4.0],
                index=2,
                format_func=lambda x: f"{x}x",
                key="playback_speed_val",
                label_visibility="collapsed"
            )

        with frame_col:
            st.markdown(f'<div class="time-display" style="text-align: right; padding-top: 10px;">Frame {int(st.session_state.current_time * fps):,}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Ensure current_time is within valid bounds
    st.session_state.current_time = max(0.0, min(st.session_state.current_time, total_duration))

    # Convert time to frame
    selected_frame = int(st.session_state.current_time * fps)
    selected_frame = min(selected_frame, total_frames - 1)

    # Re-open video to get selected frame
    if os.path.exists(video_path):
        review_cap = cv2.VideoCapture(video_path)
        review_tracker = RodentTracker(
            var_threshold=var_threshold,
            shadow_threshold=shadow_threshold,
            min_contour_area=min_contour_area
        )

        # Process frames to initialize MOG2 background model
        start_frame = max(0, selected_frame - 30)
        review_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        review_frame = None
        review_mask = None

        for i in range(start_frame, selected_frame + 1):
            ret, frame = review_cap.read()
            if not ret:
                break
            _, review_mask, _ = review_tracker.process_frame(frame, roi_pct, detect_body_parts=False)
            if i == selected_frame:
                review_frame = frame

        review_cap.release()

        # Display the selected frame
        if review_frame is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Video Feed")
                stored_centroid = (tracking_data[selected_frame]['x'], tracking_data[selected_frame]['y'])

                # Reconstruct body parts from stored data if available
                stored_body_parts = None
                if detect_body_parts and 'head_x' in tracking_data[selected_frame]:
                    frame_data = tracking_data[selected_frame]
                    if not np.isnan(frame_data.get('head_x', np.nan)):
                        stored_body_parts = BodyParts(
                            head=(frame_data['head_x'], frame_data['head_y']),
                            body=(frame_data['body_x'], frame_data['body_y']),
                            tail=(frame_data['tail_x'], frame_data['tail_y']),
                            orientation=frame_data['orientation'],
                            body_length=frame_data['body_length'],
                            confidence=frame_data['confidence']
                        )

                annotated_review = review_tracker.annotate_frame(
                    review_frame, stored_centroid, roi_pct, body_parts=stored_body_parts
                )
                annotated_review_rgb = cv2.cvtColor(annotated_review, cv2.COLOR_BGR2RGB)
                st.image(annotated_review_rgb, use_container_width=True)

            with col2:
                st.caption("Binary Mask")
                if review_mask is not None:
                    st.image(review_mask, use_container_width=True)

    # Handle playback
    if st.session_state.is_playing:
        # Sleep to control playback speed
        # Account for estimated processing overhead (~30ms)
        target_frame_time = (1.0 / fps) / playback_speed
        processing_overhead = 0.03
        sleep_time = max(0.01, target_frame_time - processing_overhead)
        time.sleep(sleep_time)
        st.rerun()

    # Create DataFrame and calculate metrics
    df = pd.DataFrame(tracking_data)
    df['velocity_cm_s'] = calculate_velocity(df, fps, pixels_per_cm)

    thigmotaxis_data = calculate_thigmotaxis(
        df, crop_width, crop_height, center_zone_pct
    )

    freezing_data = detect_freezing(
        df, fps, freezing_velocity, freezing_duration
    )

    summary_stats = generate_summary_stats(
        df, fps, pixels_per_cm,
        crop_width, crop_height,
        center_zone_pct,
        freezing_velocity,
        freezing_duration
    )

    # Display results
    display_results(
        df=df,
        summary_stats=summary_stats,
        thigmotaxis_data=thigmotaxis_data,
        freezing_data=freezing_data,
        fps=fps,
        pixels_per_cm=pixels_per_cm,
        crop_width=crop_width,
        crop_height=crop_height,
        center_zone_pct=center_zone_pct,
        freezing_velocity=freezing_velocity
    )

    # New Analysis button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start New Analysis", use_container_width=False):
        st.session_state.analysis_complete = False
        st.session_state.tracking_data = None
        st.session_state.video_path = None
        st.session_state.analysis_params = None
        st.session_state.is_playing = False
        st.session_state.current_time = 0.0
        cleanup_temp(temp_dir)
        st.rerun()


def show_welcome_screen():
    """Display welcome screen when no video is loaded."""

    # What is Movyent? expandable section
    with st.expander("What is Movyent?", expanded=False):
        st.markdown("""
        **Movyent** is a browser-based behavioral tracking platform that helps students and labs
        quantify animal movement on low-spec devices.

        #### Who It's For
        - Undergraduate and graduate students
        - Teaching labs with limited budgets
        - Research labs starting behavior experiments
        - Schools using Chromebooks

        #### What It Does
        - Track animal movement using computer vision
        - Calculate distance, velocity, and zone metrics
        - Detect freezing behavior automatically
        - Export data to CSV/Excel
        - Works 100% offline

        #### Why It Matters
        - Runs on Chromebooks and low-spec laptops
        - No GPU required
        - Free and open source
        - Reproducible research results
        - Accessible to underfunded labs

        #### How It Works
        1. **Upload Video** - Upload video of your behavioral arena
        2. **Track** - Movyent tracks the animal frame-by-frame
        3. **Analyze** - Analyze metrics and export your data

        #### FAQ
        - **What animals can I track?** Mice, rats, Drosophila, zebrafish, and other small animals with good contrast against the arena.
        - **Does it work offline?** Yes, 100% offline. Your data never leaves your computer.
        - **What file formats are supported?** MP4, AVI, and MOV video files up to 200MB.
        - **Do I need a powerful computer?** No. Movyent is optimized for Chromebooks and low-spec devices.
        - **Is it free?** Yes, Movyent is free and open source.
        """)

    # Feature cards below
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: #ffffff; padding: 1.5rem; border-radius: 12px;
                    border: 1px solid #dee2e6; height: 160px;">
            <h4 style="color: #000000; margin-bottom: 0.75rem; font-weight: 600;">MOG2 Tracking</h4>
            <p style="color: #4a5568; font-size: 0.9rem; line-height: 1.5;">
                Classical background subtraction algorithm.
                CPU-friendly, no GPU required.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #ffffff; padding: 1.5rem; border-radius: 12px;
                    border: 1px solid #dee2e6; height: 160px;">
            <h4 style="color: #000000; margin-bottom: 0.75rem; font-weight: 600;">Behavioral Metrics</h4>
            <p style="color: #4a5568; font-size: 0.9rem; line-height: 1.5;">
                Distance, velocity, thigmotaxis, freezing detection,
                zone crossings.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: #ffffff; padding: 1.5rem; border-radius: 12px;
                    border: 1px solid #dee2e6; height: 160px;">
            <h4 style="color: #000000; margin-bottom: 0.75rem; font-weight: 600;">Data Export</h4>
            <p style="color: #4a5568; font-size: 0.9rem; line-height: 1.5;">
                Export tracking data and statistics
                to CSV or Excel.
            </p>
        </div>
        """, unsafe_allow_html=True)


def show_preview(cap: cv2.VideoCapture, roi_pct: dict):
    """Show a preview of the first frame with ROI overlay."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()

    if ret:
        st.markdown("### Video Preview")

        # Draw ROI rectangle
        h, w = frame.shape[:2]
        top = int(h * roi_pct['top'] / 100)
        bottom = int(h * (100 - roi_pct['bottom']) / 100)
        left = int(w * roi_pct['left'] / 100)
        right = int(w * (100 - roi_pct['right']) / 100)

        preview = frame.copy()
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 0), 2)

        # Convert BGR to RGB for display
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(preview_rgb, caption="Green rectangle shows Region of Interest (ROI)")


def run_analysis(
    video_path: str,
    temp_dir: str,
    total_frames: int,
    fps: float,
    pixels_per_cm: float,
    roi_pct: dict,
    crop_width: int,
    crop_height: int,
    var_threshold: int,
    shadow_threshold: int,
    min_contour_area: int,
    freezing_velocity: float,
    freezing_duration: float,
    center_zone_pct: float,
    preview_interval: int,
    detect_body_parts: bool = True
):
    """Run the full tracking and analysis pipeline."""

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Initialize tracker
    tracker = RodentTracker(
        var_threshold=var_threshold,
        shadow_threshold=shadow_threshold,
        min_contour_area=min_contour_area
    )

    # Reset video position
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create UI elements
    st.markdown("### Processing")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Video Feed")
        frame_placeholder = st.empty()
    with col2:
        st.caption("Binary Mask")
        mask_placeholder = st.empty()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Storage for tracking data
    tracking_data = []

    # Process frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame with optional body part detection
        centroid, mask, body_parts = tracker.process_frame(
            frame, roi_pct, detect_body_parts=detect_body_parts
        )

        # Store data
        frame_data = {
            'frame': frame_idx,
            'x': centroid[0],
            'y': centroid[1]
        }

        # Add body part data if available
        if body_parts is not None:
            frame_data.update(body_parts.to_dict())
        else:
            # Add NaN values for body parts when not detected
            frame_data.update({
                'head_x': np.nan, 'head_y': np.nan,
                'body_x': np.nan, 'body_y': np.nan,
                'tail_x': np.nan, 'tail_y': np.nan,
                'orientation': np.nan, 'body_length': np.nan,
                'confidence': np.nan
            })

        tracking_data.append(frame_data)

        # Update preview periodically
        if frame_idx % preview_interval == 0:
            # Annotate frame with body parts
            annotated = tracker.annotate_frame(
                frame, centroid, roi_pct, body_parts=body_parts
            )
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Resize for display
            display_width = 400
            scale = display_width / annotated_rgb.shape[1]
            display_height = int(annotated_rgb.shape[0] * scale)
            annotated_resized = cv2.resize(annotated_rgb, (display_width, display_height))
            mask_resized = cv2.resize(mask, (display_width, display_height))

            frame_placeholder.image(annotated_resized)
            mask_placeholder.image(mask_resized)

            # Update progress
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Frame {frame_idx + 1:,} / {total_frames:,}")

        frame_idx += 1

    cap.release()

    # Clear preview
    progress_bar.progress(1.0)
    status_text.text("Processing complete")

    # Store results in session state
    st.session_state.analysis_complete = True
    st.session_state.tracking_data = tracking_data
    st.session_state.video_path = video_path
    st.session_state.analysis_params = {
        'fps': fps,
        'pixels_per_cm': pixels_per_cm,
        'roi_pct': roi_pct,
        'crop_width': crop_width,
        'crop_height': crop_height,
        'var_threshold': var_threshold,
        'shadow_threshold': shadow_threshold,
        'min_contour_area': min_contour_area,
        'freezing_velocity': freezing_velocity,
        'freezing_duration': freezing_duration,
        'center_zone_pct': center_zone_pct,
        'total_frames': total_frames,
        'detect_body_parts': detect_body_parts
    }

    # Rerun to show frame review with session state
    st.rerun()


def display_results(
    df: pd.DataFrame,
    summary_stats: dict,
    thigmotaxis_data: dict,
    freezing_data: dict,
    fps: float,
    pixels_per_cm: float,
    crop_width: int,
    crop_height: int,
    center_zone_pct: float,
    freezing_velocity: float
):
    """Display analysis results with visualizations and export options."""

    st.markdown("### Results")

    # Summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Distance",
            f"{summary_stats['total_distance_cm']:.1f} cm"
        )
    with col2:
        st.metric(
            "Avg Velocity",
            f"{summary_stats['avg_velocity_cm_s']:.2f} cm/s"
        )
    with col3:
        st.metric(
            "Thigmotaxis",
            f"{summary_stats['thigmotaxis_index']:.3f}"
        )
    with col4:
        st.metric(
            "Freezing",
            f"{summary_stats['freezing_percentage']:.1f}%"
        )

    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Max Velocity",
            f"{summary_stats['max_velocity_cm_s']:.2f} cm/s"
        )
    with col2:
        st.metric(
            "Zone Crossings",
            f"{summary_stats['zone_crossings']}"
        )
    with col3:
        st.metric(
            "Freezing Episodes",
            f"{freezing_data['num_episodes']}"
        )
    with col4:
        st.metric(
            "Tracking Quality",
            f"{summary_stats['tracking_quality_pct']:.1f}%"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trajectory",
        "Heatmap",
        "Velocity",
        "Zones",
        "Freezing"
    ])

    with tab1:
        fig_traj = plot_trajectory(
            df,
            arena_dims=(crop_width, crop_height),
            center_zone_pct=center_zone_pct
        )
        st.pyplot(fig_traj)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="Download PNG",
                data=fig_to_bytes(fig_traj, 'png'),
                file_name="trajectory.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="Download SVG",
                data=fig_to_bytes(fig_traj, 'svg'),
                file_name="trajectory.svg",
                mime="image/svg+xml",
                use_container_width=True
            )

    with tab2:
        fig_heat = plot_heatmap(
            df,
            arena_dims=(crop_width, crop_height),
            center_zone_pct=center_zone_pct
        )
        st.pyplot(fig_heat)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="Download PNG",
                data=fig_to_bytes(fig_heat, 'png'),
                file_name="heatmap.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="Download SVG",
                data=fig_to_bytes(fig_heat, 'svg'),
                file_name="heatmap.svg",
                mime="image/svg+xml",
                use_container_width=True
            )

    with tab3:
        fig_vel = plot_velocity_timeseries(
            df, fps,
            freezing_threshold=freezing_velocity
        )
        st.pyplot(fig_vel)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="Download PNG",
                data=fig_to_bytes(fig_vel, 'png'),
                file_name="velocity.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="Download SVG",
                data=fig_to_bytes(fig_vel, 'svg'),
                file_name="velocity.svg",
                mime="image/svg+xml",
                use_container_width=True
            )

    with tab4:
        fig_zone = plot_zone_occupancy(thigmotaxis_data, fps)
        st.pyplot(fig_zone)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="Download PNG",
                data=fig_to_bytes(fig_zone, 'png'),
                file_name="zones.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="Download SVG",
                data=fig_to_bytes(fig_zone, 'svg'),
                file_name="zones.svg",
                mime="image/svg+xml",
                use_container_width=True
            )

    with tab5:
        fig_freeze = plot_freezing_episodes(df, freezing_data, fps)
        st.pyplot(fig_freeze)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="Download PNG",
                data=fig_to_bytes(fig_freeze, 'png'),
                file_name="freezing.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="Download SVG",
                data=fig_to_bytes(fig_freeze, 'svg'),
                file_name="freezing.svg",
                mime="image/svg+xml",
                use_container_width=True
            )

    # Download all figures as ZIP
    st.markdown("<br>", unsafe_allow_html=True)
    all_figures = {
        'trajectory': fig_traj,
        'heatmap': fig_heat,
        'velocity': fig_vel,
        'zones': fig_zone,
        'freezing': fig_freeze
    }

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.download_button(
            label="Download All Figures (ZIP)",
            data=create_figures_zip(all_figures),
            file_name="movyent_figures.zip",
            mime="application/zip",
            use_container_width=True
        )

    # Data preview
    st.markdown("### Data Preview")

    export_df = prepare_export_dataframe(df, fps, pixels_per_cm)
    st.dataframe(export_df.head(100), use_container_width=True)

    if len(df) > 100:
        st.caption(f"Showing first 100 of {len(df):,} rows")

    # Export section
    st.markdown("### Export Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Raw data CSV
        csv_raw = export_df.to_csv(index=False)
        st.download_button(
            label="Results (CSV)",
            data=csv_raw,
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Summary CSV
        summary_df = pd.DataFrame([
            {'metric': k, 'value': v}
            for k, v in summary_stats.items()
        ])
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Summary (CSV)",
            data=csv_summary,
            file_name="summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        # Excel export
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name='Tracking Data', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        excel_buffer.seek(0)
        st.download_button(
            label="All Data (Excel)",
            data=excel_buffer.getvalue(),
            file_name="movyent_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


def fig_to_bytes(fig, format='png', dpi=150):
    """Convert matplotlib figure to bytes for download with white background."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf.getvalue()


def create_figures_zip(figures_dict, include_svg=True):
    """Create a ZIP file containing all figures."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figures_dict.items():
            # Add PNG
            png_data = fig_to_bytes(fig, 'png', dpi=300)
            zf.writestr(f"{name}.png", png_data)
            # Add SVG
            if include_svg:
                svg_data = fig_to_bytes(fig, 'svg')
                zf.writestr(f"{name}.svg", svg_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def cleanup_temp(temp_dir: str):
    """Clean up temporary files."""
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
