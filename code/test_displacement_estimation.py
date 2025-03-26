import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
import psutil
import gc

# Import the estimator directly since we're already in the code directory
from micromotion_estimator import ShipMicroMotionEstimator

class TestLogger:
    """Simple logger for testing that captures messages"""
    
    def __init__(self, log_file="test_displacement.log"):
        self.messages = []
        self.log_file = log_file
        
        # Clear log file if it exists
        with open(self.log_file, 'w') as f:
            f.write("Starting displacement estimation test\n")
            
    def __call__(self, message):
        """Log callback function that can be passed to the estimator"""
        self.messages.append(message)
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")


def create_synthetic_data(rows=512, cols=512, num_subapertures=7, amplitude=1.0, noise_level=0.1):
    """
    Create synthetic SAR data with controlled displacements for testing
    
    Parameters
    ----------
    rows : int
        Number of rows in each subaperture
    cols : int
        Number of columns in each subaperture
    num_subapertures : int
        Number of subapertures to create
    amplitude : float
        Amplitude of the displacement pattern
    noise_level : float
        Level of noise to add
        
    Returns
    -------
    tuple
        Tuple containing (subapertures, slc_images, ground_truth_displacements)
    """
    print(f"Creating synthetic data: {rows}x{cols}, {num_subapertures} subapertures")
    
    # Create a list to store subapertures
    subapertures = []
    
    # Create known displacement patterns between subapertures
    range_displacements = []
    azimuth_displacements = []
    
    # Base pattern with some features to track
    base_pattern = np.zeros((rows, cols), dtype=np.complex64)
    
    # Add some features (circles, rectangles) to make tracking work better
    for i in range(5):
        center_r = np.random.randint(rows // 4, 3 * rows // 4)
        center_c = np.random.randint(cols // 4, 3 * cols // 4)
        radius = np.random.randint(20, 50)
        
        # Create a circle
        r_grid, c_grid = np.ogrid[:rows, :cols]
        mask = ((r_grid - center_r)**2 + (c_grid - center_c)**2) <= radius**2
        base_pattern[mask] = 5.0 + 5.0j
    
    # Add a few rectangular features
    for i in range(3):
        top = np.random.randint(rows // 4, 3 * rows // 4)
        left = np.random.randint(cols // 4, 3 * cols // 4)
        height = np.random.randint(30, 80)
        width = np.random.randint(30, 80)
        
        base_pattern[top:top+height, left:left+width] = 7.0 + 7.0j
    
    # Create subapertures with known displacements
    for i in range(num_subapertures):
        # Start with the base pattern
        subap = base_pattern.copy()
        
        # Add random noise
        subap += noise_level * (np.random.normal(0, 1, (rows, cols)) + 
                              1j * np.random.normal(0, 1, (rows, cols)))
        
        # Apply displacement (if not the first subaperture)
        if i > 0:
            # Create a displacement field with a simple pattern
            range_shift = amplitude * np.sin(2 * np.pi * i / num_subapertures)
            azimuth_shift = amplitude * np.cos(2 * np.pi * i / num_subapertures)
            
            # Store the ground truth displacement
            range_displacements.append(range_shift)
            azimuth_displacements.append(azimuth_shift)
            
            # Apply shift using numpy roll (integer shift only for simplicity)
            subap = np.roll(subap, int(range_shift * 5), axis=0)
            subap = np.roll(subap, int(azimuth_shift * 5), axis=1)
        
        subapertures.append(subap)
    
    # Create SLC images from subapertures (just use magnitude)
    slc_images = [np.abs(subap) for subap in subapertures]
    
    return subapertures, slc_images, (range_displacements, azimuth_displacements)


def test_displacement_estimation(rows=512, cols=512, num_subapertures=7, 
                                window_size=64, overlap=0.5, memory_efficient=True):
    """
    Test the displacement estimation function
    
    Parameters
    ----------
    rows : int
        Number of rows in each subaperture
    cols : int
        Number of columns in each subaperture
    num_subapertures : int
        Number of subapertures to create
    window_size : int
        Window size for displacement estimation
    overlap : float
        Overlap ratio for windows
    memory_efficient : bool
        Whether to use the memory-efficient version
        
    Returns
    -------
    bool
        True if the test passed, False otherwise
    dict
        Dictionary with test results
    """
    # Set up logging
    logger = TestLogger()
    
    # Create output directory for test results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic data
    subapertures, slc_images, ground_truth = create_synthetic_data(
        rows=rows, cols=cols, num_subapertures=num_subapertures
    )
    
    # Create estimator
    estimator = ShipMicroMotionEstimator(
        num_subapertures=num_subapertures,
        window_size=window_size,
        overlap=overlap,
        debug_mode=True,
        log_callback=logger
    )
    
    # Inject synthetic data
    estimator.subapertures = subapertures
    estimator.slc_images = slc_images
    
    # Measure memory before processing
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    logger(f"Memory usage before displacement estimation: {memory_before:.2f} MB")
    
    # Start time measurement
    start_time = time.time()
    
    # Run displacement estimation
    success = False
    error_message = None
    displacement_maps = None
    
    try:
        if memory_efficient:
            logger("Testing enhanced memory-efficient displacement estimation...")
            success = estimator.estimate_displacement_enhanced()
        else:
            logger("Testing standard displacement estimation...")
            success = estimator.estimate_displacement()
            
        displacement_maps = estimator.displacement_maps
        
        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        logger(f"Memory usage after displacement estimation: {memory_after:.2f} MB")
        logger(f"Memory change: {memory_after - memory_before:.2f} MB")
    
    except Exception as e:
        error_message = str(e)
        trace = traceback.format_exc()
        logger(f"Error during displacement estimation: {error_message}")
        logger(trace)
    
    # End time measurement
    end_time = time.time()
    duration = end_time - start_time
    
    # Force garbage collection
    gc.collect()
    
    # Validate results
    validation_result = validate_results(
        estimator, ground_truth, success, error_message, duration, 
        rows, cols, window_size, overlap, memory_efficient
    )
    
    # Save results
    results = {
        "success": success,
        "error_message": error_message,
        "duration": duration,
        "validation": validation_result,
        "parameters": {
            "rows": rows,
            "cols": cols,
            "num_subapertures": num_subapertures,
            "window_size": window_size,
            "overlap": overlap,
            "memory_efficient": memory_efficient
        }
    }
    
    # Plot results if successful
    if success and displacement_maps:
        plot_results(estimator, ground_truth, output_dir, memory_efficient)
    
    return success, results


def validate_results(estimator, ground_truth, success, error_message, duration,
                    rows, cols, window_size, overlap, memory_efficient):
    """
    Validate the displacement estimation results against ground truth
    
    Parameters
    ----------
    estimator : ShipMicroMotionEstimator
        The estimator with displacement maps
    ground_truth : tuple
        Tuple containing (range_displacements, azimuth_displacements)
    success : bool
        Whether the displacement estimation was successful
    error_message : str
        Error message if not successful
    duration : float
        Duration of the displacement estimation in seconds
    rows, cols, window_size, overlap : int, int, int, float
        Parameters used for the test
    memory_efficient : bool
        Whether the memory-efficient version was used
        
    Returns
    -------
    dict
        Dictionary with validation results
    """
    validation = {
        "execution_time": duration,
        "success": success,
        "error": error_message,
        "mse_range": None,
        "mse_azimuth": None,
        "maps_created": False,
        "maps_valid": False,
        "shape_correct": False
    }
    
    if not success:
        return validation
    
    # Check if displacement maps were created
    if not hasattr(estimator, 'displacement_maps') or estimator.displacement_maps is None:
        validation["maps_created"] = False
        return validation
    
    validation["maps_created"] = True
    
    # Check number of displacement maps
    expected_maps = len(estimator.slc_images) - 1
    if len(estimator.displacement_maps) != expected_maps:
        validation["maps_valid"] = False
        return validation
    
    validation["maps_valid"] = True
    
    # Check shape of displacement maps
    step = int(window_size * (1 - overlap))
    expected_rows = (rows - window_size) // step + 1
    expected_cols = (cols - window_size) // step + 1
    
    for range_map, azimuth_map in estimator.displacement_maps:
        if range_map.shape != (expected_rows, expected_cols) or \
           azimuth_map.shape != (expected_rows, expected_cols):
            validation["shape_correct"] = False
            return validation
    
    validation["shape_correct"] = True
    
    # Compare to ground truth displacements - this is approximate since the
    # displacement estimation is done over windows
    ground_truth_range, ground_truth_azimuth = ground_truth
    
    # Calculate mean displacements from maps
    mean_range_disps = []
    mean_azimuth_disps = []
    
    for range_map, azimuth_map in estimator.displacement_maps:
        # Filter out zeros and extreme values
        valid_range = range_map[(range_map != 0) & (np.abs(range_map) < 10)]
        valid_azimuth = azimuth_map[(azimuth_map != 0) & (np.abs(azimuth_map) < 10)]
        
        if len(valid_range) > 0:
            mean_range_disps.append(np.mean(valid_range))
        else:
            mean_range_disps.append(0)
            
        if len(valid_azimuth) > 0:
            mean_azimuth_disps.append(np.mean(valid_azimuth))
        else:
            mean_azimuth_disps.append(0)
    
    # Calculate MSE against ground truth
    if len(mean_range_disps) == len(ground_truth_range) and len(ground_truth_range) > 0:
        mse_range = np.mean((np.array(mean_range_disps) - np.array(ground_truth_range))**2)
        validation["mse_range"] = mse_range
        
    if len(mean_azimuth_disps) == len(ground_truth_azimuth) and len(ground_truth_azimuth) > 0:
        mse_azimuth = np.mean((np.array(mean_azimuth_disps) - np.array(ground_truth_azimuth))**2)
        validation["mse_azimuth"] = mse_azimuth
    
    return validation


def plot_results(estimator, ground_truth, output_dir, memory_efficient):
    """
    Plot the displacement estimation results and compare to ground truth
    
    Parameters
    ----------
    estimator : ShipMicroMotionEstimator
        The estimator with displacement maps
    ground_truth : tuple
        Tuple containing (range_displacements, azimuth_displacements)
    output_dir : str
        Directory to save output plots
    memory_efficient : bool
        Whether the memory-efficient version was used
    """
    if estimator.displacement_maps is None or len(estimator.displacement_maps) == 0:
        return
    
    # Extract ground truth
    ground_truth_range, ground_truth_azimuth = ground_truth
    
    # Create figure for displacement maps
    plt.figure(figsize=(15, 10))
    
    # Plot the first displacement map
    range_map, azimuth_map = estimator.displacement_maps[0]
    
    plt.subplot(2, 2, 1)
    plt.imshow(range_map, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Range Displacement Map (First Pair)')
    
    plt.subplot(2, 2, 2)
    plt.imshow(azimuth_map, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Azimuth Displacement Map (First Pair)')
    
    # Calculate mean displacements for comparison with ground truth
    mean_range_disps = []
    mean_azimuth_disps = []
    
    for range_map, azimuth_map in estimator.displacement_maps:
        # Filter out zeros and extreme values
        valid_range = range_map[(range_map != 0) & (np.abs(range_map) < 10)]
        valid_azimuth = azimuth_map[(azimuth_map != 0) & (np.abs(azimuth_map) < 10)]
        
        if len(valid_range) > 0:
            mean_range_disps.append(np.mean(valid_range))
        else:
            mean_range_disps.append(0)
            
        if len(valid_azimuth) > 0:
            mean_azimuth_disps.append(np.mean(valid_azimuth))
        else:
            mean_azimuth_disps.append(0)
    
    # Compare estimated displacements with ground truth
    plt.subplot(2, 2, 3)
    plt.plot(mean_range_disps, 'bo-', label='Estimated')
    if len(ground_truth_range) > 0:
        plt.plot(ground_truth_range, 'ro-', label='Ground Truth')
    plt.xlabel('Subaperture Pair')
    plt.ylabel('Mean Range Displacement')
    plt.legend()
    plt.title('Range Displacement Comparison')
    
    plt.subplot(2, 2, 4)
    plt.plot(mean_azimuth_disps, 'bo-', label='Estimated')
    if len(ground_truth_azimuth) > 0:
        plt.plot(ground_truth_azimuth, 'ro-', label='Ground Truth')
    plt.xlabel('Subaperture Pair')
    plt.ylabel('Mean Azimuth Displacement')
    plt.legend()
    plt.title('Azimuth Displacement Comparison')
    
    plt.tight_layout()
    
    # Save the plot
    version = "enhanced" if memory_efficient else "standard"
    plt.savefig(os.path.join(output_dir, f"displacement_test_{version}.png"))
    plt.close()


def run_all_tests():
    """
    Run a comprehensive set of tests with different parameters
    """
    print("Starting comprehensive displacement estimation tests")
    
    # List of test configurations
    test_configs = [
        # Format: (rows, cols, num_subapertures, window_size, overlap, memory_efficient)
        # Small test cases
        (128, 128, 5, 32, 0.5, True),
        (128, 128, 5, 32, 0.5, False),
        
        # Medium test cases
        (512, 512, 7, 64, 0.5, True),
        (512, 512, 7, 64, 0.5, False),
        
        # Large test case (only with memory-efficient version)
        (1024, 1024, 7, 64, 0.5, True),
        
        # Very large test case (only with memory-efficient version)
        (2048, 2048, 7, 128, 0.5, True),
    ]
    
    # Results storage
    results = []
    
    # Run each test configuration
    for config in test_configs:
        rows, cols, num_subapertures, window_size, overlap, memory_efficient = config
        
        print("\n" + "="*80)
        print(f"Testing {'memory-efficient' if memory_efficient else 'standard'} displacement estimation")
        print(f"Parameters: rows={rows}, cols={cols}, num_subapertures={num_subapertures}")
        print(f"            window_size={window_size}, overlap={overlap}")
        print("="*80)
        
        # Run the test
        success, result = test_displacement_estimation(
            rows=rows, cols=cols, num_subapertures=num_subapertures,
            window_size=window_size, overlap=overlap, memory_efficient=memory_efficient
        )
        
        # Store the result
        results.append({
            "config": config,
            "success": success,
            "result": result
        })
        
        # Clean up
        gc.collect()
        
        print("\nTest complete:", "SUCCESS" if success else "FAILURE")
        print(f"Duration: {result['duration']:.2f} seconds")
        if result['error_message']:
            print(f"Error: {result['error_message']}")
        print("-"*80 + "\n")
    
    # Summarize results
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for i, res in enumerate(results):
        config = res["config"]
        rows, cols, num_subapertures, window_size, overlap, memory_efficient = config
        version = "Memory-Efficient" if memory_efficient else "Standard"
        print(f"{i+1}. {version} ({rows}x{cols}, {num_subapertures} subapertures): ", end="")
        if res["success"]:
            print(f"SUCCESS ({res['result']['duration']:.2f} seconds)")
        else:
            print(f"FAILURE - {res['result']['error_message']}")
    
    # Determine if any test failed
    all_successful = all(res["success"] for res in results)
    
    print("\nOVERALL RESULT:", "ALL TESTS PASSED" if all_successful else "SOME TESTS FAILED")
    return all_successful


if __name__ == "__main__":
    # Run all tests
    all_successful = run_all_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if all_successful else 1) 