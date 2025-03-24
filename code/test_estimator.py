import os
import numpy as np
import matplotlib.pyplot as plt
from micromotion_estimator import ShipMicroMotionEstimator

def create_synthetic_data(rows=512, cols=512, num_subapertures=7):
    """
    Create synthetic SAR data for testing the micro-motion estimator
    
    Parameters
    ----------
    rows : int, optional
        Number of rows in the data (default is 512)
    cols : int, optional
        Number of columns in the data (default is 512)
    num_subapertures : int, optional
        Number of sub-apertures to create (default is 7)
        
    Returns
    -------
    tuple
        Tuple containing (subapertures, pvp, ship_locations)
    """
    # Create synthetic phase history data
    data = np.zeros((rows, cols), dtype=np.complex64)
    
    # Add background noise
    data += np.random.normal(0, 0.1, (rows, cols)) + 1j * np.random.normal(0, 0.1, (rows, cols))
    
    # Add synthetic ships with different vibration patterns
    ship_locations = []
    
    # Ship 1: Strong vibration at 15 Hz
    ship1_row, ship1_col = 150, 200
    ship_locations.append((ship1_row, ship1_col))
    data[ship1_row-20:ship1_row+20, ship1_col-30:ship1_col+30] += 5.0
    
    # Ship 2: Multiple vibration modes (10 Hz and 25 Hz)
    ship2_row, ship2_col = 300, 350
    ship_locations.append((ship2_row, ship2_col))
    data[ship2_row-25:ship2_row+25, ship2_col-40:ship2_col+40] += 7.0
    
    # Create synthetic PVP data (placeholder)
    pvp = {
        'RANGE_TIME': np.linspace(0, 1, rows),
        'AZIMUTH_TIME': np.linspace(0, 1, cols)
    }
    
    # Create synthetic sub-apertures with vibration effects
    subapertures = []
    for i in range(num_subapertures):
        subap = data.copy()
        
        # Add time-varying displacement to simulate vibration for Ship 1 (15 Hz)
        # This creates a more pronounced effect that should be detectable
        ship1_range_shift = 0.5 * np.sin(2 * np.pi * 15 * i / num_subapertures)
        ship1_azimuth_shift = 0.3 * np.cos(2 * np.pi * 15 * i / num_subapertures)
        
        # Apply the shift to the ship region using a more direct approach
        ship1_region = subap[ship1_row-20:ship1_row+20, ship1_col-30:ship1_col+30]
        shifted_region = np.roll(ship1_region, int(ship1_range_shift * 5), axis=0)
        shifted_region = np.roll(shifted_region, int(ship1_azimuth_shift * 5), axis=1)
        subap[ship1_row-20:ship1_row+20, ship1_col-30:ship1_col+30] = shifted_region
        
        # Add time-varying displacement to simulate vibration for Ship 2 (10 Hz and 25 Hz)
        ship2_range_shift = 0.4 * np.sin(2 * np.pi * 10 * i / num_subapertures) + 0.2 * np.sin(2 * np.pi * 25 * i / num_subapertures)
        ship2_azimuth_shift = 0.3 * np.cos(2 * np.pi * 10 * i / num_subapertures) + 0.15 * np.cos(2 * np.pi * 25 * i / num_subapertures)
        
        # Apply the shift to the ship region
        ship2_region = subap[ship2_row-25:ship2_row+25, ship2_col-40:ship2_col+40]
        shifted_region = np.roll(ship2_region, int(ship2_range_shift * 5), axis=0)
        shifted_region = np.roll(shifted_region, int(ship2_azimuth_shift * 5), axis=1)
        subap[ship2_row-25:ship2_row+25, ship2_col-40:ship2_col+40] = shifted_region
        
        subapertures.append(subap)
    
    return subapertures, pvp, ship_locations

def test_with_synthetic_data():
    """
    Test the micro-motion estimator with synthetic data
    """
    print("Creating synthetic data...")
    subapertures, pvp, ship_locations = create_synthetic_data()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Testing ShipMicroMotionEstimator with synthetic data...")
    
    # Create estimator
    estimator = ShipMicroMotionEstimator(num_subapertures=7, window_size=64, overlap=0.5)
    
    # Override the load_data and _focus_subapertures methods for testing
    estimator.subapertures = subapertures
    
    # Generate SLC images with more pronounced features to help with displacement estimation
    estimator.slc_images = []
    for subap in subapertures:
        # Use magnitude of the complex data as the SLC image
        slc = np.abs(subap)
        # Enhance contrast to make features more distinct
        slc = np.power(slc, 0.5)  # Square root to enhance contrast
        estimator.slc_images.append(slc)
    
    # Estimate displacement
    print("Estimating displacement...")
    if estimator.estimate_displacement():
        # Use the known ship locations as measurement points
        print("Analyzing time series at ship locations...")
        if estimator.analyze_time_series(ship_locations):
            # Plot results for each measurement point
            for i in range(len(ship_locations)):
                print(f"Plotting results for ship {i+1}...")
                estimator.plot_results(i, output_dir=output_dir)
                
                # Identify dominant frequencies
                dominant_freqs = estimator.identify_dominant_frequencies(i, threshold=0.3)
                if dominant_freqs is not None:
                    print(f"\nDominant frequencies for ship {i+1}:")
                    print("Range:")
                    for freq, amp in dominant_freqs['range']:
                        print(f"  {freq:.2f} Hz (amplitude: {amp:.2f})")
                    print("Azimuth:")
                    for freq, amp in dominant_freqs['azimuth']:
                        print(f"  {freq:.2f} Hz (amplitude: {amp:.2f})")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    test_with_synthetic_data()
