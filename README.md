# Ship Micro-Motion Estimation from SAR Images

This application implements the algorithm described in the paper "Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in Cosmo-Skymed Synthetic Aperture Radar Data—An Operative Assessment" (https://www.mdpi.com/2072-4292/11/14/1637).

## Overview

The application estimates micro-motion of ships from Synthetic Aperture Radar (SAR) images using pixel tracking techniques. It can detect vibrations and oscillations in different parts of ships, which can be used for ship characterization and identification.

## Features

- Load and process CPHD (Complex Phase History Data) files
- Split SAR data into multiple Doppler sub-apertures
- Estimate displacement between adjacent sub-apertures using sub-pixel offset tracking
- Analyze time series of displacements at measurement points
- Identify dominant vibration frequencies through spectral analysis
- Visualize results with time series plots and frequency spectra

## Requirements

- Python 3.10+
- Flask
- NumPy
- SciPy
- Matplotlib
- scikit-image
- SarPy (for reading CPHD files)

## Installation

```bash
pip install flask numpy scipy matplotlib scikit-image sarpy
```

## Usage

### Running the Web Application

```bash
cd ship_micromotion
python code/app.py
```

The application will be available at http://localhost:5000

### Using the API

```python
from micromotion_estimator import ShipMicroMotionEstimator

# Create estimator
estimator = ShipMicroMotionEstimator(num_subapertures=7, window_size=64, overlap=0.5)

# Load data
if estimator.load_data('path/to/your/cphd/file.cphd'):
    # Estimate displacement
    if estimator.estimate_displacement():
        # Define measurement points
        measurement_points = [(100, 100), (200, 200), (300, 300)]
        
        # Analyze time series
        if estimator.analyze_time_series(measurement_points):
            # Plot results for each measurement point
            for i in range(len(measurement_points)):
                estimator.plot_results(i, output_dir="results")
                
                # Identify dominant frequencies
                dominant_freqs = estimator.identify_dominant_frequencies(i)
                if dominant_freqs is not None:
                    print(f"\nDominant frequencies for measurement point {i}:")
                    print("Range:")
                    for freq, amp in dominant_freqs['range']:
                        print(f"  {freq:.2f} Hz (amplitude: {amp:.2f})")
                    print("Azimuth:")
                    for freq, amp in dominant_freqs['azimuth']:
                        print(f"  {freq:.2f} Hz (amplitude: {amp:.2f})")
```

## Algorithm Details

The algorithm follows these key steps:

1. **Sub-aperture Creation**: The SAR data is split into multiple Doppler sub-apertures, which represent the same scene at slightly different times.

2. **SLC Image Generation**: Each sub-aperture is focused to generate a Single Look Complex (SLC) image.

3. **Displacement Estimation**: Sub-pixel offset tracking is used to measure displacements between adjacent sub-apertures. This is done using phase cross-correlation.

4. **Time Series Analysis**: The displacement time series at specified measurement points are analyzed to detect oscillations.

5. **Frequency Analysis**: Spectral analysis is performed to identify dominant vibration frequencies.

## Interpreting Results

- **Time Series Plots**: Show the displacement of measurement points over time in both range and azimuth directions. Oscillations indicate vibration or micro-motion.

- **Frequency Spectra**: Show the dominant frequencies present in the time series. Peaks correspond to vibration modes.

- **Typical Interpretation**:
  - Low frequencies (1-5 Hz): Usually correspond to overall ship motion (roll, pitch, yaw)
  - Mid frequencies (5-20 Hz): Often related to main structural vibrations
  - High frequencies (20+ Hz): Typically associated with machinery, engines, or smaller structural elements

## Data Sources

The application can work with CPHD data from various sources, including the Umbra open data catalog:
http://umbra-open-data-catalog.s3-website.us-west-2.amazonaws.com/?prefix=sar-data/tasks/ship_detection_testdata/0c4a34d4-671d-412f-a8c5-fcb7543fd220/2023-08-31-01-09-38_UMBRA-04/

## License

This project is licensed under the MIT License - see the LICENSE file for details.
