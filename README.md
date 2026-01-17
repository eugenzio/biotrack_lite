# Movyent

**Behavioral Analysis System | MOG2 Background Subtraction**

A web-based rodent tracking and behavioral analysis platform designed for neuroscience research. Movyent uses classical computer vision (MOG2 background subtraction) to track animal movement and generate publication-quality metrics and visualizations.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)

## 🚀 [Try Movyent Now - No Installation Required!](https://movyent.streamlit.app/)

Click the link above to use Movyent directly in your browser without downloading or installing anything.

---

## 🎯 Key Features

### 🎥 Video Analysis
- **Supported Formats**: MP4, AVI, MOV
- **Automatic FPS Detection**: Smart detection of video frame rate
- **ROI Cropping**: Flexible region-of-interest selection to exclude arena walls
- **Real-time Preview**: Visual feedback during analysis with adjustable preview intervals

### 🐭 Animal Tracking
- **MOG2 Background Subtraction**: Explainable classical CV algorithm ideal for research publications
- **Shadow Handling**: Advanced shadow detection and removal for accurate tracking
- **Body Part Detection**: Estimates head, body center, and tail positions using contour geometry
- **Orientation Tracking**: Calculates animal orientation (0-360°) and body length

### 📊 Behavioral Metrics
- **Velocity Analysis**: Frame-by-frame velocity calculation (cm/s)
- **Total Distance**: Cumulative distance traveled throughout experiment
- **Thigmotaxis**: Center vs. periphery time and distance ratios
- **Freezing Detection**: Configurable freezing episodes based on velocity threshold and duration
- **Zone Crossings**: Center zone entry/exit event counting

### 📈 Visualization Suite
All plots are publication-quality and colorblind-friendly:

1. **Trajectory Plot**: Animal path colored by velocity
2. **Heatmap**: Spatial occupancy density with center zone overlay
3. **Velocity Timeseries**: Velocity over time with freezing episodes highlighted
4. **Zone Occupancy**: Bar chart comparing center vs. periphery time
5. **Freezing Episodes**: Timeline visualization of freezing behavior
6. **Summary Dashboard**: Multi-panel overview of all key metrics

### 📥 Export Capabilities
- **Excel Export**: Complete tracking data with all metrics (`.xlsx`)
- **CSV Export**: Raw tracking data for custom analysis
- **Figure Export**:
  - Individual plots (PNG 300 DPI, SVG)
  - Batch download as ZIP archive
  - High-resolution images for publications

### 🎮 Interactive Media Player
- **Playback Controls**: Play/Pause, Restart, ±5s skip
- **Speed Control**: 0.25x to 4x playback speed
- **Frame-by-Frame Navigation**: Precise frame stepping
- **Time Slider**: Visual timeline with current position indicator

---

## 🚀 Quick Start

### Option 1: Use Online (Recommended)

**No installation required!** Simply visit:

### **[https://movyent.streamlit.app/](https://movyent.streamlit.app/)**

Upload your video and start analyzing immediately in your browser.

---

### Option 2: Run Locally

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movyent.git
   cd movyent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

#### Running Locally

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

### Option 3: Deploy Your Own Instance

Want to deploy your own version on Streamlit Cloud?

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy by connecting your forked repository
4. The app will use `requirements.txt` and `runtime.txt` automatically
5. Share your custom deployment URL with collaborators

---

## 📖 Usage Guide

### 1. Upload Video
- Click "Upload Video" and select your video file (MP4, AVI, or MOV)
- The app will automatically detect video properties (FPS, dimensions, duration)

### 2. Configure Settings

**Calibration & ROI**
- **FPS**: Use detected FPS or enter manually for accurate time-based metrics
- **Arena Width**: Enter the actual arena width in centimeters for distance calibration
- **ROI Cropping**: Adjust sliders to crop out walls or non-arena areas

**Tracking Parameters**
- **MOG2 Variance**: Lower values = more sensitive background detection (default: 16)
- **Shadow Threshold**: Adjust shadow removal sensitivity (default: 250)
- **Min Contour Area**: Minimum pixel area to consider as the animal (default: 500)
- **Freezing Velocity**: Velocity threshold for freezing detection (default: 2.0 cm/s)
- **Freezing Duration**: Minimum time to count as freezing episode (default: 1.0 s)
- **Center Zone**: Percentage of arena considered "center" for thigmotaxis (default: 60%)
- **Body Parts Detection**: Enable/disable head/body/tail estimation

### 3. Run Analysis
- Click "Run Analysis" to start tracking
- Monitor progress bar for real-time feedback
- Preview frames appear at specified intervals

### 4. Review Results
- Explore interactive visualizations
- Review summary statistics table
- Use media player to verify tracking accuracy

### 5. Export Data
- Download tracking data as Excel or CSV
- Export individual figures or download all as ZIP
- Choose PNG (300 DPI) or SVG formats

---

## 🏗️ Project Structure

```
movyent/
├── app.py                      # Main Streamlit application
├── tracker.py                  # RodentTracker class (MOG2 implementation)
├── metrics.py                  # Behavioral metrics calculation
├── visualization.py            # Publication-quality plotting functions
├── behavior_classifier.py      # Optional behavior classification
├── pose_tracker.py            # Optional YOLO pose estimation
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version for Streamlit Cloud
├── packages.txt              # System packages for deployment
└── README.md                 # This file
```

### Core Modules

**`tracker.py`**
- `RodentTracker`: Main tracking class using MOG2 background subtraction
- `BodyParts`: Data class for storing head/body/tail positions
- Implements shadow handling and morphological noise filtering

**`metrics.py`**
- `calculate_velocity()`: Frame-by-frame velocity computation
- `calculate_total_distance()`: Cumulative distance traveled
- `calculate_thigmotaxis()`: Center vs. periphery analysis
- `detect_freezing()`: Freezing episode detection
- `count_zone_crossings()`: Center zone entry/exit events
- `generate_summary_stats()`: Aggregated statistics table
- `prepare_export_dataframe()`: Data formatting for export

**`visualization.py`**
- `plot_trajectory()`: Velocity-colored path visualization
- `plot_heatmap()`: Spatial occupancy density map
- `plot_velocity_timeseries()`: Velocity over time with freezing overlay
- `plot_zone_occupancy()`: Center vs. periphery bar chart
- `plot_freezing_episodes()`: Freezing timeline visualization
- `plot_summary_dashboard()`: Multi-panel overview figure

---

## 🧪 Algorithm Details

### MOG2 Background Subtraction

Movyent uses **Mixture of Gaussians (MOG2)** for background subtraction, a classical computer vision algorithm that:

1. **Learns Background**: Models each pixel as a mixture of Gaussian distributions
2. **Detects Foreground**: Identifies pixels that deviate from the background model
3. **Adapts Over Time**: Updates the model continuously to handle lighting changes
4. **Handles Shadows**: Marks shadows as gray (127) which are then removed

**Why MOG2?**
- ✅ **Explainable**: Unlike deep learning, you can explain the algorithm in a research paper
- ✅ **No Training Required**: Works out-of-the-box without labeled data
- ✅ **Lightweight**: Runs on low-spec computers and Chromebooks
- ✅ **Reliable**: Proven algorithm used in hundreds of neuroscience publications

### Body Part Detection

The algorithm estimates head, body, and tail positions using:

1. **Contour Detection**: Finds the animal's outline in the foreground mask
2. **Ellipse Fitting**: Fits an ellipse to determine orientation (major axis)
3. **Movement Direction**: Analyzes velocity to disambiguate head from tail
4. **Geometric Calculation**: Estimates head/tail positions along the major axis

**Note**: This is a geometric approximation, not true pose estimation. For precise keypoint tracking, 

---

## ⚙️ Configuration

### System Requirements

**Minimum**:
- Python 3.11+
- 4GB RAM
- Modern web browser (Chrome recommended)

**Recommended**:
- Python 3.11
- 8GB+ RAM
- SSD storage for faster video loading

### Dependencies

Core dependencies (see `requirements.txt`):
- `streamlit` - Web application framework
- `opencv-python-headless` - Computer vision (headless for cloud deployment)
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `openpyxl` - Excel file export

Optional:
- `ultralytics` - YOLOv8 pose estimation (if using `pose_tracker.py`)

### Environment Variables

For Streamlit Cloud deployment, you can configure:

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 1000  # Max video size in MB
enableXsrfProtection = true

[theme]
primaryColor = "#2563eb"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#000000"
```

---

## 🐛 Troubleshooting

### Common Issues

**Video won't upload**
- Check file size (Streamlit Cloud has ~200MB limit by default)
- Ensure video format is MP4, AVI, or MOV
- Try converting to H.264 codec MP4

**Tracking is inaccurate**
- Adjust MOG2 Variance (lower = more sensitive)
- Increase Min Contour Area to filter out small noise
- Use ROI cropping to exclude arena walls
- Ensure static camera (MOG2 requires fixed background)

**Freezing detection is wrong**
- Adjust Freezing Velocity threshold (lower = more sensitive)
- Increase Freezing Duration to avoid false positives
- Verify Arena Width is calibrated correctly

**App crashes on Streamlit Cloud**
- Ensure `opencv-python-headless` is used (not `opencv-python`)
- Check `runtime.txt` specifies Python 3.11
- Add system dependencies to `packages.txt` if needed

**Slow performance**
- Reduce video resolution before uploading
- Increase Preview Interval to reduce UI updates
- Disable Body Parts Detection if not needed

---

## 🎓 Citation

If you use Movyent in your research, please cite:

```bibtex
@software{movyent2025,
  title = {Movyent: Behavioral Analysis System for Rodent Tracking},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/movyent}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (if applicable)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to new functions
- Keep functions focused and modular

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenCV** - Computer vision foundation
- **Streamlit** - Web application framework
- **Matplotlib/Seaborn** - Visualization libraries
- Neuroscience community for feedback and feature requests

---

## 📧 Contact

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Email: eugene201903t@gmail.com

---

<div align="center">

**Designed for low-spec computers and Chromebooks**

![Chrome Compatible](https://img.shields.io/badge/Chrome-Compatible-green?style=flat&logo=googlechrome)

Made with ❤️ for the neuroscience community

</div>
