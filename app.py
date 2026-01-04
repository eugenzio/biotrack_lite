"""
app.py - Professional Research Dashboard for BioTrack-Lite

This is the main entry point for the BioTrack-Lite application.
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
from tracker import RodentTracker
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


# Professional Gray Theme CSS
PROFESSIONAL_THEME = """
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Main app background */
    .stApp {
        background-color: #1e1e1e;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #252526;
        border-right: 1px solid #3c3c3c;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #cccccc;
    }

    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #cccccc !important;
    }

    /* Headers */
    h1 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.02em;
        border-bottom: 2px solid #3c3c3c;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
    }

    h2, h3 {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }

    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid #3c3c3c;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }

    /* Text color */
    .stMarkdown, p, span, label {
        color: #d4d4d4;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #2d2d2d 0%, #252526 100%);
        border: 1px solid #3c3c3c;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    [data-testid="stMetric"] label {
        color: #808080 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0e639c 0%, #1177bb 100%);
        color: white !important;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1177bb 0%, #1e90ff 100%);
        box-shadow: 0 2px 8px rgba(14, 99, 156, 0.4);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0e639c 0%, #1177bb 100%);
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2d2d2d 0%, #3c3c3c 100%);
        border: 1px solid #4a4a4a;
        color: #ffffff !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #3c3c3c 0%, #4a4a4a 100%);
        border-color: #5a5a5a;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #252526;
        border-radius: 8px 8px 0 0;
        padding: 0.25rem;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #808080;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3c3c3c;
        color: #ffffff;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0e639c !important;
        color: #ffffff !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: #252526;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
        border: 1px solid #3c3c3c;
        border-top: none;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #2d2d2d;
        border: 2px dashed #4a4a4a;
        border-radius: 8px;
        padding: 1rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #0e639c;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        background-color: #3c3c3c;
    }

    .stSlider [data-testid="stThumbValue"] {
        color: #ffffff;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #0e639c;
    }

    /* Info/Success/Error boxes */
    .stAlert {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        border-radius: 6px;
    }

    [data-testid="stAlert"] {
        background-color: #2d2d2d;
        border-left: 4px solid #0e639c;
    }

    /* Success message */
    .stSuccess {
        background-color: #1e3a29 !important;
        border-left-color: #28a745 !important;
    }

    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        background-color: #252526;
        border-radius: 8px;
    }

    .dataframe {
        background-color: #252526 !important;
        color: #d4d4d4 !important;
    }

    .dataframe th {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    .dataframe td {
        background-color: #252526 !important;
        color: #d4d4d4 !important;
        border-color: #3c3c3c !important;
    }

    /* Caption text */
    .stCaption {
        color: #808080 !important;
        font-size: 0.75rem;
    }

    /* Divider */
    hr {
        border-color: #3c3c3c;
    }

    /* Image containers */
    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #3c3c3c;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        border-radius: 6px;
        color: #ffffff !important;
    }

    /* Number input */
    .stNumberInput input {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        color: #ffffff;
        border-radius: 4px;
    }

    /* Checkbox */
    .stCheckbox label span {
        color: #d4d4d4 !important;
    }

    /* Column containers */
    [data-testid="column"] {
        padding: 0.25rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }

    ::-webkit-scrollbar-thumb {
        background: #3c3c3c;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #4a4a4a;
    }
</style>
"""


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="BioTrack-Lite | Behavioral Analysis",
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
        <h1 style="margin-bottom: 0.5rem;">BioTrack-Lite</h1>
        <p style="color: #808080;">
            Offline Behavioral Analysis System | MOG2 Background Subtraction
        </p>
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
        <div style="background: #2d2d2d; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
            <div style="color: #28a745; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                Video Loaded
            </div>
            <div style="color: #808080;">
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
                preview_interval=preview_interval
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

    # Frame Review Section
    st.markdown("### Video Review")

    # Calculate total duration
    total_duration = total_frames / fps

    # Helper function to format time as MM:SS
    def format_time(seconds):
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"

    # Playback controls
    col_play, col_speed, col_time = st.columns([1, 1, 2])

    with col_play:
        # Play/Pause button
        if st.session_state.is_playing:
            if st.button("⏸ Pause", use_container_width=True, key="pause_btn"):
                st.session_state.is_playing = False
                st.rerun()
        else:
            if st.button("▶ Play", use_container_width=True, key="play_btn"):
                st.session_state.is_playing = True
                st.rerun()

    with col_speed:
        playback_speed = st.selectbox(
            "Speed",
            options=[0.25, 0.5, 1.0, 2.0, 4.0],
            index=2,
            format_func=lambda x: f"{x}x",
            key="playback_speed"
        )

    with col_time:
        st.markdown(f"""
        <div style="padding: 0.5rem; background: #2d2d2d; border-radius: 6px; text-align: center;">
            <span style="color: #ffffff; font-size: 1.2rem; font-family: monospace;">
                {format_time(st.session_state.current_time)} / {format_time(total_duration)}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Time slider (in seconds)
    selected_time = st.slider(
        "Time",
        min_value=0.0,
        max_value=total_duration,
        value=st.session_state.current_time,
        step=1.0 / fps,
        format="%.2f s",
        key="time_slider",
        label_visibility="collapsed"
    )

    # Update current_time if slider moved manually
    if not st.session_state.is_playing:
        st.session_state.current_time = selected_time

    # Convert time to frame
    selected_frame = int(selected_time * fps)
    selected_frame = min(selected_frame, total_frames - 1)

    # Display frame info
    st.caption(f"Frame {selected_frame:,} / {total_frames - 1:,}")

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
            _, review_mask = review_tracker.process_frame(frame, roi_pct)
            if i == selected_frame:
                review_frame = frame

        review_cap.release()

        # Display the selected frame
        if review_frame is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Video Feed")
                stored_centroid = (tracking_data[selected_frame]['x'], tracking_data[selected_frame]['y'])
                annotated_review = review_tracker.annotate_frame(review_frame, stored_centroid, roi_pct)
                annotated_review_rgb = cv2.cvtColor(annotated_review, cv2.COLOR_BGR2RGB)
                st.image(annotated_review_rgb, use_container_width=True)

            with col2:
                st.caption("Binary Mask")
                if review_mask is not None:
                    st.image(review_mask, use_container_width=True)

    # Handle playback
    if st.session_state.is_playing:
        # Calculate next time step based on playback speed
        time_step = (1.0 / fps) * playback_speed * 3  # Multiply by 3 for smoother playback
        next_time = st.session_state.current_time + time_step

        if next_time >= total_duration:
            # Stop at end
            st.session_state.is_playing = False
            st.session_state.current_time = 0.0
        else:
            st.session_state.current_time = next_time

        # Small delay for visual update
        time.sleep(0.05)
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

    # Feature cards below
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: #252526; padding: 1.5rem; border-radius: 8px;
                    border: 1px solid #3c3c3c; height: 160px;">
            <h4 style="color: #ffffff; margin-bottom: 0.75rem;">MOG2 Tracking</h4>
            <p style="color: #808080; font-size: 0.9rem;">
                Classical background subtraction algorithm.
                CPU-friendly, no GPU required.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #252526; padding: 1.5rem; border-radius: 8px;
                    border: 1px solid #3c3c3c; height: 160px;">
            <h4 style="color: #ffffff; margin-bottom: 0.75rem;">Behavioral Metrics</h4>
            <p style="color: #808080; font-size: 0.9rem;">
                Distance, velocity, thigmotaxis, freezing detection,
                zone crossings.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: #252526; padding: 1.5rem; border-radius: 8px;
                    border: 1px solid #3c3c3c; height: 160px;">
            <h4 style="color: #ffffff; margin-bottom: 0.75rem;">Data Export</h4>
            <p style="color: #808080; font-size: 0.9rem;">
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
    preview_interval: int
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

        # Process frame
        centroid, mask = tracker.process_frame(frame, roi_pct)

        # Store data
        tracking_data.append({
            'frame': frame_idx,
            'x': centroid[0],
            'y': centroid[1]
        })

        # Update preview periodically
        if frame_idx % preview_interval == 0:
            # Annotate frame
            annotated = tracker.annotate_frame(frame, centroid, roi_pct)
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
        'total_frames': total_frames
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
            file_name="biotrack_figures.zip",
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
            file_name="biotrack_results.xlsx",
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
