import os
import numpy as np
import matplotlib.pyplot as plt
from micromotion_estimator import ShipMicroMotionEstimator
from ship_region_detector import ShipRegionDetector

def create_synthetic_data(rows=512, cols=512, num_subapertures=7, ship_locations=None):
    """
    Create synthetic data for testing the ship micro-motion estimation algorithm
    
    Parameters
    ----------
    rows : int, optional
        Number of rows in the image (default is 512)
    cols : int, optional
        Number of columns in the image (default is 512)
    num_subapertures : int, optional
        Number of sub-apertures to create (default is 7)
    ship_locations : list, optional
        List of (row, col, width, height) tuples specifying ship locations
        
    Returns
    -------
    tuple
        Tuple containing (subapertures, pvp, ship_masks)
    """
    # Default ship locations if none provided
    if ship_locations is None:
        ship_locations = [
            (150, 200, 30, 120),  # Ship 1: (row, col, width, height)
            (300, 350, 40, 100)   # Ship 2: (row, col, width, height)
        ]
    
    # Create subapertures
    subapertures = []
    ship_masks = []
    
    # Create base image with ships
    base_image = np.zeros((rows, cols), dtype=np.complex128)
    
    # Add ships to base image
    for i, (row, col, width, height) in enumerate(ship_locations):
        # Create ship mask
        ship_mask = np.zeros((rows, cols), dtype=bool)
        r_min = max(0, row - height // 2)
        r_max = min(rows, row + height // 2)
        c_min = max(0, col - width // 2)
        c_max = min(cols, col + width // 2)
        ship_mask[r_min:r_max, c_min:c_max] = True
        
        # Add ship to base image with higher intensity
        base_image[ship_mask] = 10.0 + 5.0j
        
        # Add some texture to the ship
        texture = np.random.normal(0, 1, (r_max - r_min, c_max - c_min)) + \
                 1j * np.random.normal(0, 1, (r_max - r_min, c_max - c_min))
        base_image[r_min:r_max, c_min:c_max] += texture
        
        # Store ship mask
        ship_masks.append(ship_mask)
    
    # Add some speckle to the background
    background_mask = ~np.logical_or.reduce(ship_masks) if ship_masks else np.ones((rows, cols), dtype=bool)
    base_image[background_mask] = np.random.normal(0, 0.1, np.sum(background_mask)) + \
                                 1j * np.random.normal(0, 0.1, np.sum(background_mask))
    
    # Create subapertures with micro-motion
    for i in range(num_subapertures):
        # Copy base image
        subap = base_image.copy()
        
        # Add micro-motion to ships
        for j, (row, col, width, height) in enumerate(ship_locations):
            # Define vibration parameters for each ship
            if j == 0:
                # Ship 1: Strong vibration at 15 Hz
                freq = 15.0
                amp_range = 0.01
                amp_azimuth = 0.02
                phase_range = 0.0
                phase_azimuth = np.pi / 2
            else:
                # Ship 2: Multiple vibration modes (10 Hz and 25 Hz)
                freq1 = 10.0
                freq2 = 25.0
                amp_range = 0.01
                amp_azimuth = 0.015
                phase_range = np.pi / 4
                phase_azimuth = np.pi / 3
            
            # Calculate time for this subaperture
            t = i / (num_subapertures - 1) if num_subapertures > 1 else 0
            
            # Calculate displacement for Ship 1
            if j == 0:
                disp_range = amp_range * np.sin(2 * np.pi * freq * t + phase_range)
                disp_azimuth = amp_azimuth * np.sin(2 * np.pi * freq * t + phase_azimuth)
            # Calculate displacement for Ship 2 (multiple frequencies)
            else:
                disp_range = amp_range * (np.sin(2 * np.pi * freq1 * t + phase_range) + 
                                         0.5 * np.sin(2 * np.pi * freq2 * t + phase_range + np.pi/4))
                disp_azimuth = amp_azimuth * (np.sin(2 * np.pi * freq1 * t + phase_azimuth) + 
                                             0.5 * np.sin(2 * np.pi * freq2 * t + phase_azimuth + np.pi/3))
            
            # Apply displacement to ship
            r_min = max(0, row - height // 2)
            r_max = min(rows, row + height // 2)
            c_min = max(0, col - width // 2)
            c_max = min(cols, col + width // 2)
            
            # Create a copy of the ship region
            ship_region = subap[r_min:r_max, c_min:c_max].copy()
            
            # Clear the original ship region
            subap[r_min:r_max, c_min:c_max] = 0
            
            # Calculate new coordinates with displacement
            new_r_min = int(r_min + disp_azimuth)
            new_r_max = int(r_max + disp_azimuth)
            new_c_min = int(c_min + disp_range)
            new_c_max = int(c_max + disp_range)
            
            # Ensure new coordinates are within image bounds
            new_r_min = max(0, new_r_min)
            new_r_max = min(rows, new_r_max)
            new_c_min = max(0, new_c_min)
            new_c_max = min(cols, new_c_max)
            
            # Calculate the valid region sizes
            valid_height = min(new_r_max - new_r_min, r_max - r_min)
            valid_width = min(new_c_max - new_c_min, c_max - c_min)
            
            # Place the ship at the new position
            if valid_height > 0 and valid_width > 0:
                subap[new_r_min:new_r_min+valid_height, new_c_min:new_c_min+valid_width] = \
                    ship_region[:valid_height, :valid_width]
        
        # Add the subaperture to the list
        subapertures.append(subap)
    
    # Create dummy PVP data
    pvp = {
        'time': np.linspace(0, 1, num_subapertures),
        'position': np.zeros((num_subapertures, 3)),
        'velocity': np.zeros((num_subapertures, 3))
    }
    
    return subapertures, pvp, ship_masks

def test_vibration_energy_visualization():
    """
    Test the vibration energy visualization functionality
    """
    print("Testing vibration energy visualization...")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic data with more pronounced ships
    ship_locations = [
        (150, 200, 40, 150),  # Ship 1: (row, col, width, height)
        (300, 350, 50, 120)   # Ship 2: (row, col, width, height)
    ]
    subapertures, pvp, ship_masks = create_synthetic_data(
        rows=512, cols=512, num_subapertures=10, ship_locations=ship_locations
    )
    
    # Create estimator
    estimator = ShipMicroMotionEstimator(num_subapertures=10, window_size=32, overlap=0.5)
    
    # Set subapertures and generate SLC images
    estimator.subapertures = subapertures
    estimator.slc_images = []
    for subap in subapertures:
        # Use magnitude of the complex data as the SLC image
        slc = np.abs(subap)
        # Enhance contrast to make features more distinct
        slc = np.power(slc, 0.5)  # Square root to enhance contrast
        estimator.slc_images.append(slc)
    
    # Estimate displacement
    print("Estimating displacement...")
    estimator.estimate_displacement()
    
    # Calculate vibration energy map
    print("Calculating vibration energy map...")
    estimator.calculate_vibration_energy_map()
    
    # Detect ship regions
    print("Detecting ship regions...")
    estimator.detect_ship_regions(num_regions=3, energy_threshold=-15)
    
    # Plot vibration energy map
    print("Plotting vibration energy map...")
    estimator.plot_vibration_energy_map(output_dir=output_dir)
    
    # Use the ShipRegionDetector for more advanced detection
    print("Using ShipRegionDetector for advanced region detection...")
    detector = ShipRegionDetector(min_region_size=20, energy_threshold=-15, num_regions=3)
    ship_regions = detector.detect_regions(estimator.vibration_energy_map_db)
    
    # Get region statistics
    print("Calculating region statistics...")
    region_stats = detector.get_region_statistics(
        estimator.vibration_energy_map_db, estimator.displacement_maps
    )
    
    # Print region statistics
    print("\nShip Region Statistics:")
    for stats in region_stats:
        print(f"Region {stats['id']}:")
        print(f"  Centroid: ({stats['centroid'][0]:.1f}, {stats['centroid'][1]:.1f})")
        print(f"  Area: {stats['area']} pixels")
        print(f"  Mean Energy: {stats['mean_energy']:.2f} dB")
        print(f"  Max Energy: {stats['max_energy']:.2f} dB")
        if 'mean_range_disp' in stats:
            print(f"  Mean Range Displacement: {stats['mean_range_disp']:.4f} pixels")
            print(f"  Mean Azimuth Displacement: {stats['mean_azimuth_disp']:.4f} pixels")
        print()
    
    # Create a more detailed visualization with region statistics
    print("Creating detailed visualization with region statistics...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot SLC image on the left
    if estimator.slc_images is not None and len(estimator.slc_images) > 0:
        # Use the first SLC image
        slc_image = estimator.slc_images[0]
        axs[0].imshow(slc_image, cmap='gray')
        axs[0].set_title('SLC Image with Ship Regions')
        axs[0].set_xlabel('Range (pixels)')
        axs[0].set_ylabel('Azimuth (pixels)')
    
    # Plot vibration energy map on the right
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=-25, vmax=0)
    im = axs[1].imshow(estimator.vibration_energy_map_db, cmap=cmap, norm=norm)
    axs[1].set_title('SLC ROI Vibration Energy (dB)')
    axs[1].set_xlabel('Range (pixels)')
    axs[1].set_ylabel('Azimuth (pixels)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axs[1])
    cbar.set_label('Vibration Energy (dB)')
    
    # Add ship region labels and annotations
    for region in detector.ship_regions:
        region_id = region['id']
        centroid = region['centroid']
        
        # Find corresponding statistics
        stats = next((s for s in region_stats if s['id'] == region_id), None)
        
        # Add label to SLC image
        axs[0].text(centroid[1], centroid[0], str(region_id), 
                   color='white', fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Add label to vibration energy map
        axs[1].text(centroid[1], centroid[0], str(region_id), 
                   color='white', fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Add arrows pointing to the regions
        arrow_length = 40
        arrow_angle = np.random.uniform(0, 2*np.pi)  # Random angle for variety
        arrow_dx = arrow_length * np.cos(arrow_angle)
        arrow_dy = arrow_length * np.sin(arrow_angle)
        arrow_start_x = centroid[1] + arrow_dx
        arrow_start_y = centroid[0] + arrow_dy
        
        # For SLC image
        axs[0].annotate('', xy=(centroid[1], centroid[0]), 
                       xytext=(arrow_start_x, arrow_start_y),
                       arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
        
        # For vibration energy map
        axs[1].annotate('', xy=(centroid[1], centroid[0]), 
                       xytext=(arrow_start_x, arrow_start_y),
                       arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
        
        # Add statistics annotation
        if stats:
            stats_text = f"Region {region_id}:\n"
            stats_text += f"Energy: {stats['mean_energy']:.1f} dB\n"
            if 'mean_range_disp' in stats:
                stats_text += f"Range Disp: {stats['mean_range_disp']:.3f} px\n"
                stats_text += f"Azimuth Disp: {stats['mean_azimuth_disp']:.3f} px"
            
            # Position the text box near the arrow start
            text_x = arrow_start_x + 10 * np.cos(arrow_angle)
            text_y = arrow_start_y + 10 * np.sin(arrow_angle)
            
            # Add text to both plots
            axs[0].annotate(stats_text, xy=(text_x, text_y), 
                           color='white', fontsize=8,
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
            
            axs[1].annotate(stats_text, xy=(text_x, text_y), 
                           color='white', fontsize=8,
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_vibration_energy_map.png'))
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_dir, 'detailed_vibration_energy_map.png')}")
    return True

if __name__ == "__main__":
    test_vibration_energy_visualization()
