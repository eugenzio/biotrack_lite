"""
app.py - Streamlit Dashboard for BioTrack-Lite

This is the main entry point for the BioTrack-Lite application.
It provides a web-based interface for video analysis, parameter tuning,
and result export.

Run with: streamlit run app.py

Design Philosophy:
- Memory efficient: Process frames one at a time, never store full video
- Interactive: Real-time preview during processing
- Export-ready: Generate publication-quality CSV files
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

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


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="BioTrack-Lite",
        page_icon="🐭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("🐭 BioTrack-Lite")
    st.markdown("""
    **Offline Behavioral Analysis Tool for Frugal Neuroscience**

    Analyze pre-recorded videos of rodents in Open Field Tests using
    classical computer vision (MOG2 background subtraction).
    """)

    # Sidebar configuration
    st.sidebar.header("📁 Video Input")

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Video",
        type=['mp4', 'avi', 'mov'],
        help="Supported formats: MP4, AVI, MOV"
    )

    if uploaded_file is None:
        st.info("👆 Upload a video file to begin analysis.")
        show_instructions()
        return

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Open video and get properties
    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():
        st.error("❌ Could not open video file. Please try a different format.")
        cleanup_temp(temp_dir)
        return

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Display video info
    st.sidebar.success(f"✅ Video loaded: {total_frames} frames")

    # FPS configuration
    st.sidebar.header("⏱️ Frame Rate")
    use_detected_fps = st.sidebar.checkbox(
        f"Use detected FPS ({detected_fps:.2f})",
        value=True if detected_fps > 0 else False
    )

    if use_detected_fps and detected_fps > 0:
        fps = detected_fps
    else:
        fps = st.sidebar.number_input(
            "Manual FPS",
            min_value=1.0,
            max_value=120.0,
            value=30.0,
            step=1.0,
            help="Enter the video frame rate manually"
        )

    # Calibration
    st.sidebar.header("📏 Calibration")
    arena_width_cm = st.sidebar.number_input(
        "Arena Width (cm)",
        min_value=1.0,
        max_value=500.0,
        value=40.0,
        step=1.0,
        help="Physical width of the arena for distance calculations"
    )

    # ROI Cropping
    st.sidebar.header("✂️ ROI Cropping")
    st.sidebar.caption("Crop edges to remove timestamps, cage bars, etc.")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        roi_top = st.slider("Top %", 0, 30, 0, key="roi_top")
        roi_left = st.slider("Left %", 0, 30, 0, key="roi_left")
    with col2:
        roi_bottom = st.slider("Bottom %", 0, 30, 0, key="roi_bottom")
        roi_right = st.slider("Right %", 0, 30, 0, key="roi_right")

    roi_pct = {
        'top': roi_top,
        'bottom': roi_bottom,
        'left': roi_left,
        'right': roi_right
    }

    # Calculate cropped dimensions
    crop_width = int(frame_width * (100 - roi_left - roi_right) / 100)
    crop_height = int(frame_height * (100 - roi_top - roi_bottom) / 100)

    # Pixels per cm (based on cropped width)
    pixels_per_cm = crop_width / arena_width_cm

    st.sidebar.caption(f"Effective resolution: {crop_width}×{crop_height} px")
    st.sidebar.caption(f"Scale: {pixels_per_cm:.2f} px/cm")

    # Tracking Parameters
    st.sidebar.header("🔧 Tracking Parameters")

    var_threshold = st.sidebar.slider(
        "MOG2 Variance Threshold",
        min_value=1,
        max_value=100,
        value=16,
        help="Lower = more sensitive, Higher = less sensitive"
    )

    shadow_threshold = st.sidebar.slider(
        "Shadow Removal Threshold",
        min_value=200,
        max_value=255,
        value=250,
        help="Threshold to distinguish foreground from shadows"
    )

    min_contour_area = st.sidebar.slider(
        "Min Contour Area (px)",
        min_value=100,
        max_value=2000,
        value=500,
        help="Minimum blob size to be considered the animal"
    )

    # Freezing Detection
    st.sidebar.header("❄️ Freezing Detection")

    freezing_velocity = st.sidebar.slider(
        "Velocity Threshold (cm/s)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Max velocity to be considered immobile"
    )

    freezing_duration = st.sidebar.slider(
        "Min Duration (s)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Minimum duration for a freezing episode"
    )

    # Zone Analysis
    st.sidebar.header("🎯 Zone Analysis")

    center_zone_pct = st.sidebar.slider(
        "Center Zone (%)",
        min_value=40,
        max_value=80,
        value=60,
        help="Percentage of arena considered 'center zone'"
    ) / 100.0

    # Process button
    st.sidebar.header("🚀 Analysis")

    preview_interval = st.sidebar.slider(
        "Preview Interval",
        min_value=1,
        max_value=30,
        value=10,
        help="Update preview every N frames (higher = faster processing)"
    )

    if st.sidebar.button("▶️ Start Analysis", type="primary", use_container_width=True):
        run_analysis(
            cap=cap,
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
    else:
        # Show preview of first frame
        show_preview(cap, roi_pct)
        cap.release()
        cleanup_temp(temp_dir)


def show_instructions():
    """Display usage instructions when no video is loaded."""
    st.markdown("""
    ### How to Use BioTrack-Lite

    1. **Upload a video** of your Open Field Test (MP4, AVI, or MOV)
    2. **Configure parameters** in the sidebar:
       - Set the correct FPS if auto-detection fails
       - Enter your arena dimensions for calibration
       - Adjust ROI cropping to remove artifacts
       - Tune tracking sensitivity if needed
    3. **Click "Start Analysis"** to process the video
    4. **Review results** and download CSV files

    ### Requirements
    - Single animal in a static arena
    - Consistent lighting (no shadows moving independently)
    - Camera positioned directly above the arena

    ### Technical Details
    BioTrack-Lite uses **MOG2 Background Subtraction**, a classical
    computer vision algorithm that models the background as a mixture
    of Gaussian distributions. This approach is:
    - **Explainable**: You can describe exactly how it works
    - **CPU-friendly**: No GPU required
    - **Robust**: Works with various arena types and lighting conditions
    """)


def show_preview(cap: cv2.VideoCapture, roi_pct: dict):
    """Show a preview of the first frame with ROI overlay."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()

    if ret:
        st.subheader("📷 Video Preview")

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
        st.image(preview_rgb, caption="Green rectangle shows the Region of Interest (ROI)")


def run_analysis(
    cap: cv2.VideoCapture,
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

    # Initialize tracker
    tracker = RodentTracker(
        var_threshold=var_threshold,
        shadow_threshold=shadow_threshold,
        min_contour_area=min_contour_area
    )

    # Reset video position
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create UI elements
    st.subheader("🔄 Processing Video")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original Frame (with centroid)")
        frame_placeholder = st.empty()
    with col2:
        st.caption("Binary Mask (computer view)")
        mask_placeholder = st.empty()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Storage for tracking data (memory efficient: only coordinates)
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
            status_text.text(f"Processing frame {frame_idx + 1}/{total_frames}")

        frame_idx += 1

    cap.release()

    # Clear preview
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")

    # Create DataFrame
    df = pd.DataFrame(tracking_data)

    # Calculate metrics
    st.subheader("📊 Calculating Metrics")

    df['velocity_cm_s'] = calculate_velocity(df, fps, pixels_per_cm)

    thigmotaxis_data = calculate_thigmotaxis(
        df, crop_width, crop_height, center_zone_pct
    )

    freezing_data = detect_freezing(
        df, fps, freezing_velocity, freezing_duration
    )

    zone_crossings = count_zone_crossings(
        df, crop_width, crop_height, center_zone_pct
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

    # Cleanup
    cleanup_temp(temp_dir)


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

    st.subheader("📈 Results")

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
            "Thigmotaxis Index",
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

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📍 Trajectory",
        "🔥 Heatmap",
        "📈 Velocity",
        "🥧 Zones",
        "❄️ Freezing"
    ])

    with tab1:
        fig = plot_trajectory(
            df,
            arena_dims=(crop_width, crop_height),
            center_zone_pct=center_zone_pct
        )
        st.pyplot(fig)

    with tab2:
        fig = plot_heatmap(
            df,
            arena_dims=(crop_width, crop_height),
            center_zone_pct=center_zone_pct
        )
        st.pyplot(fig)

    with tab3:
        fig = plot_velocity_timeseries(
            df, fps,
            freezing_threshold=freezing_velocity
        )
        st.pyplot(fig)

    with tab4:
        fig = plot_zone_occupancy(thigmotaxis_data, fps)
        st.pyplot(fig)

    with tab5:
        fig = plot_freezing_episodes(df, freezing_data, fps)
        st.pyplot(fig)

    # Data preview
    st.subheader("📋 Data Preview")

    export_df = prepare_export_dataframe(df, fps, pixels_per_cm)
    st.dataframe(export_df.head(100), use_container_width=True)

    if len(df) > 100:
        st.caption(f"Showing first 100 of {len(df)} rows")

    # Export section
    st.subheader("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Raw data CSV
        csv_raw = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download results.csv",
            data=csv_raw,
            file_name="results.csv",
            mime="text/csv",
            help="Frame-by-frame tracking data with velocity"
        )

    with col2:
        # Summary CSV
        summary_df = pd.DataFrame([
            {'metric': k, 'value': v}
            for k, v in summary_stats.items()
        ])
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download summary.csv",
            data=csv_summary,
            file_name="summary.csv",
            mime="text/csv",
            help="Aggregated behavioral metrics"
        )


def cleanup_temp(temp_dir: str):
    """Clean up temporary files."""
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
