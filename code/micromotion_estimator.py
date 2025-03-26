import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import shift
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from skimage.registration import phase_cross_correlation
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from skimage.transform import warp, AffineTransform
from cphd_reader import split_aperture

class ShipMicroMotionEstimator:
    """
    Class for estimating micro-motion of ships from SAR images using pixel tracking
    Based on the paper: "Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in 
    Cosmo-Skymed Synthetic Aperture Radar Data—An Operative Assessment"
    """
    
    def __init__(self, num_subapertures=7, window_size=64, overlap=0.5, debug_mode=False, log_callback=None):
        """
        Initialize the micro-motion estimator
        
        Parameters
        ----------
        num_subapertures : int, optional
            Number of sub-apertures to create (default is 7)
        window_size : int, optional
            Size of the window for cross-correlation (default is 64)
        overlap : float, optional
            Overlap between adjacent windows (default is 0.5)
        debug_mode : bool, optional
            Whether to enable debug mode with extra visualization and step-by-step processing (default is False)
        log_callback : callable, optional
            A callback function that will be called with log messages (default is None)
        """
        self.num_subapertures = num_subapertures
        self.window_size = window_size
        self.overlap = overlap
        self.debug_mode = debug_mode
        self.log_callback = log_callback
        self.subapertures = None
        self.displacement_maps = None
        self.time_series = None
        self.frequency_spectra = None
        self.vibration_energy_map = None
        self.ship_regions = None
        self.data_loaded = False
        
        # Debug mode attributes
        self.raw_data = None
        self.raw_data_image = None
        self.slc_images = None
        self.snr_maps = {}
        self.coherence_maps = {}
        
        self.last_error = None  # Store the last error message and traceback
        
    def log(self, message):
        """
        Log a message with the callback if available, otherwise print.
        
        Parameters:
        - message: str, the message to log
        """
        try:
            # Ensure the message is a string
            message_str = str(message)
            
            # Print to console for immediate feedback
            print(message_str)
            
            # If a callback is provided (e.g., to update UI), use it
            if self.log_callback:
                try:
                    self.log_callback(message_str)
                except Exception as e:
                    print(f"Error in log callback: {e}")
            
            # In debug mode, also log to a file
            if self.debug_mode:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                try:
                    with open("debug_estimator.log", "a") as f:
                        f.write(f"{timestamp} - {message_str}\n")
                except Exception as e:
                    print(f"Error writing to debug log file: {e}")
                    
        except Exception as e:
            # Failsafe to ensure logging never crashes the program
            print(f"Error in logging: {e}")
        
    def load_data(self, cphd_file_path, channel_index=0):
        """
        Load CPHD data, split into sub-apertures, and focus them.

        Parameters:
        - cphd_file_path: str, path to CPHD file or 'demo' for synthetic data
        - channel_index: int, channel to process (default: 0)

        Returns:
        - bool, True if successful
        """
        if cphd_file_path == 'demo' or not os.path.exists(cphd_file_path):
            self.log("Creating synthetic demo data")
            from test_estimator import create_synthetic_data
            self.subapertures, pvp, _ = create_synthetic_data(
                rows=512, cols=512, num_subapertures=self.num_subapertures
            )
            self.subaperture_times = np.linspace(0, 1, self.num_subapertures)
            self.slc_images = [np.abs(subap) for subap in self.subapertures]
            return True
        else:
            try:
                from sarpy.io import open as sarpy_open
                
                self.log(f"Loading CPHD file: {cphd_file_path}")
                # Use sarpy's generic open function which is more robust
                reader = sarpy_open(cphd_file_path)
                reader_type = type(reader).__name__
                self.log(f"Reader type: {reader_type}")
                
                # Get metadata
                if hasattr(reader, 'cphd_meta'):
                    metadata = reader.cphd_meta
                    self.log("CPHD metadata available")
                else:
                    metadata = {}
                    self.log("Warning: No CPHD metadata found")
                
                # Read the data using read_chip which we know works based on our analysis
                if hasattr(reader, 'read_chip'):
                    self.log("Reading CPHD data using read_chip...")
                    data = reader.read_chip()
                    self.log(f"Data shape: {data.shape}")
                else:
                    raise ValueError("Reader does not have read_chip method")
                
                # For CPHDReader1, we need to provide an index to read_pvp_array
                if reader_type == 'CPHDReader1':
                    self.log("Using CPHDReader1-specific PVP reading...")
                    
                    # Create a synthetic PVP dictionary with timing info
                    if hasattr(reader, 'data_size') and len(reader.data_size) > 0:
                        num_vectors = reader.data_size[0]
                        self.log(f"Creating synthetic timing for {num_vectors} vectors")
                        
                        # Get channel information
                        if hasattr(metadata, 'Data') and hasattr(metadata.Data, 'to_dict'):
                            data_dict = metadata.Data.to_dict()
                            channels = data_dict.get('Channels', [])
                            if channels:
                                channel_info = channels[min(channel_index, len(channels)-1)]
                                if isinstance(channel_info, dict) and 'NumVectors' in channel_info:
                                    num_vectors = channel_info['NumVectors']
                        
                        # Attempt to read a single PVP array to get its structure
                        try:
                            # Get the first PVP array
                            pvp_sample = reader.read_pvp_array(0)
                            self.log(f"Read sample PVP entry: {pvp_sample.keys() if isinstance(pvp_sample, dict) else 'Not a dictionary'}")
                            
                            # Create a full PVP dictionary
                            pvp = {}
                            # Add essential TxTime field for timing
                            pvp['TxTime'] = np.linspace(0, 1, num_vectors)
                            
                            # Copy other fields from the sample if present
                            if isinstance(pvp_sample, dict):
                                for key in pvp_sample:
                                    if key != 'TxTime' and not key in pvp:
                                        # Use the same value for all vectors (approximation)
                                        pvp[key] = np.full(num_vectors, pvp_sample[key])
                        except Exception as e:
                            self.log(f"Error reading PVP sample: {e}")
                            # Create minimal pvp dictionary if we can't read a sample
                            pvp = {'TxTime': np.linspace(0, 1, num_vectors)}
                    else:
                        # Fall back to default size if we can't determine from reader
                        num_vectors = data.shape[0] if isinstance(data, np.ndarray) and len(data.shape) > 0 else 512
                        pvp = {'TxTime': np.linspace(0, 1, num_vectors)}
                else:
                    # For other reader types, try standard PVP reading methods
                    try:
                        if hasattr(reader, 'read_pvp'):
                            self.log("Reading PVP data using read_pvp...")
                            pvp = reader.read_pvp()
                        elif hasattr(reader, 'read_pvp_array'):
                            self.log("Reading PVP data using read_pvp_array...")
                            pvp = reader.read_pvp_array()
                        else:
                            raise ValueError("No suitable PVP reading method found")
                    except Exception as e:
                        self.log(f"Error reading PVP data: {e}")
                        # Create synthetic PVP with timing information
                        num_vectors = data.shape[0] if isinstance(data, np.ndarray) and len(data.shape) > 0 else 512
                        pvp = {'TxTime': np.linspace(0, 1, num_vectors)}
                
                # Split data into subapertures
                self.log("Splitting data into sub-apertures...")
                self.subapertures, self.subaperture_times = self._split_aperture(
                    data, self.num_subapertures, self.overlap
                )
                self.log(f"Created {len(self.subapertures)} sub-apertures")
                
                # Focus sub-apertures
                self.log("Focusing sub-apertures...")
                self.slc_images = self._focus_subapertures(self.subapertures, metadata)
                self.log(f"Generated {len(self.slc_images)} SLC images")
                
                # Store raw data for inspection
                self.raw_data = data
                self.pvp = pvp
                self.raw_data_image = np.abs(data)
                self.data_loaded = True
                
                return True
            except Exception as e:
                self.log(f"Error loading CPHD data: {e}")
                import traceback
                trace = traceback.format_exc()
                self.log(trace)
                self.last_error = f"{str(e)}\n\n{trace}"
                return False
    
    def _split_aperture(self, data, num_subapertures, overlap=0.5):
        """
        Split phase history data into overlapping sub-apertures based on azimuth pulses.

        Parameters:
        - data: numpy.ndarray, phase history data (pulses x samples)
        - num_subapertures: int, number of sub-apertures to create
        - overlap: float, overlap ratio between sub-apertures (0 to 1, default: 0.5)

        Returns:
        - list of numpy.ndarray, each containing a sub-aperture's data
        - list of floats, timing information for each sub-aperture
        """
        num_pulses, num_samples = data.shape
        if num_subapertures <= 0 or num_subapertures > num_pulses:
            raise ValueError("Invalid number of sub-apertures")
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")

        subaperture_size = num_pulses // num_subapertures
        step = int(subaperture_size * (1 - overlap))
        subapertures = []
        
        # Create synthetic timing for subapertures if pvp is available
        if hasattr(self, 'pvp') and self.pvp and 'TxTime' in self.pvp:
            times = self.pvp['TxTime']
            subaperture_times = []
        else:
            # Create synthetic timing
            times = np.linspace(0, 1, num_pulses)
            subaperture_times = []

        for i in range(num_subapertures):
            start_idx = i * step
            end_idx = start_idx + subaperture_size
            if end_idx > num_pulses:
                end_idx = num_pulses
                start_idx = max(0, end_idx - subaperture_size)
            subaperture = data[start_idx:end_idx, :].copy()  # Copy to avoid modifying original
            subapertures.append(subaperture)
            # Calculate average time for this subaperture
            if len(times) > 0:
                subaperture_times.append(np.mean(times[start_idx:end_idx]))
            else:
                subaperture_times.append(i / num_subapertures)

        return subapertures, subaperture_times
    
    def _focus_subapertures(self, subapertures, metadata=None):
        """Apply focusing algorithms to the subapertures to generate SLC images"""
        if not subapertures:
            self.log("No subapertures to focus")
            return []
        
        self.log(f"Focusing {len(subapertures)} subapertures")
        
        # In this simplified implementation, we just use the magnitude of each subaperture
        # as the SLC image. In a real application, this would involve actual SAR focusing
        # algorithms.
        slc_images = []
        for i, subap in enumerate(subapertures):
            # Check if subap is complex
            self.log(f"Focusing subaperture {i+1}/{len(subapertures)}")
            
            # Simply use magnitude for now
            focused = np.abs(subap)
            slc_images.append(focused)
        
        self.log(f"Generated {len(slc_images)} SLC images")
        return slc_images
    
    def estimate_displacement(self):
        """
        Estimate displacement between adjacent sub-apertures using sub-pixel offset tracking
        
        Returns
        -------
        bool
            True if displacement was estimated successfully, False otherwise
        """
        self.log("Starting displacement estimation...")
        
        if self.slc_images is None or len(self.slc_images) < 2:
            self.log("Error: No SLC images available for displacement estimation")
            return False
        
        # Initialize displacement maps
        self.log(f"Initializing displacement maps for {len(self.slc_images)-1} image pairs")
        self.displacement_maps = []
        
        # Calculate step size based on window size and overlap
        step = int(self.window_size * (1 - self.overlap))
        self.log(f"Using window size: {self.window_size}, overlap: {self.overlap}, step size: {step}")
        
        # For each pair of adjacent sub-apertures
        for i in range(len(self.slc_images) - 1):
            self.log(f"Processing image pair {i+1}/{len(self.slc_images)-1}...")
            
            ref_image = self.slc_images[i]
            sec_image = self.slc_images[i + 1]
            
            # Get image dimensions
            rows, cols = ref_image.shape
            self.log(f"Image dimensions: {rows}x{cols}")
            
            # Initialize displacement map for this pair
            range_shifts = np.zeros((rows // step, cols // step))
            azimuth_shifts = np.zeros((rows // step, cols // step))
            self.log(f"Displacement map dimensions: {range_shifts.shape}")
            
            # Check for valid image data
            if np.isnan(ref_image).any() or np.isinf(ref_image).any():
                self.log(f"Warning: Reference image contains NaN or Inf values")
            if np.isnan(sec_image).any() or np.isinf(sec_image).any():
                self.log(f"Warning: Secondary image contains NaN or Inf values")
                
            # Validate image statistics
            ref_min, ref_max = np.min(ref_image), np.max(ref_image)
            sec_min, sec_max = np.min(sec_image), np.max(sec_image)
            self.log(f"Reference image range: {ref_min:.4f} to {ref_max:.4f}")
            self.log(f"Secondary image range: {sec_min:.4f} to {sec_max:.4f}")
            
            # Count windows to process
            total_windows = ((rows - self.window_size) // step + 1) * ((cols - self.window_size) // step + 1)
            self.log(f"Processing {total_windows} windows...")
            
            window_count = 0
            error_count = 0
            
            try:
                # For each window position
                for r in range(0, rows - self.window_size, step):
                    for c in range(0, cols - self.window_size, step):
                        window_count += 1
                        if window_count % 100 == 0:
                            self.log(f"Processed {window_count}/{total_windows} windows ({window_count/total_windows*100:.1f}%)")
                            
                        # Extract windows from both images
                        ref_window = ref_image[r:r+self.window_size, c:c+self.window_size]
                        sec_window = sec_image[r:r+self.window_size, c:c+self.window_size]
                        
                        # Check window validity more thoroughly
                        if np.all(ref_window == 0) or np.all(sec_window == 0):
                            # Skip empty windows
                            continue
                            
                        # Check window contrast - skip windows with very low variance
                        ref_std = np.std(ref_window)
                        sec_std = np.std(sec_window)
                        if ref_std < 1e-3 or sec_std < 1e-3:
                            # Window has almost no signal, skip it
                            continue
                            
                        # Check for NaN or Inf values
                        if np.isnan(ref_window).any() or np.isnan(sec_window).any() or \
                           np.isinf(ref_window).any() or np.isinf(sec_window).any():
                            # Skip windows with invalid values
                            continue
                            
                        # Calculate sub-pixel shift using phase cross-correlation
                        try:
                            # Direct execution instead of threading to better catch errors
                            self.log(f"Running phase cross-correlation at position ({r}, {c})")
                            try:
                                shift, error, diffphase = phase_cross_correlation(
                                    ref_window, sec_window, upsample_factor=100
                                )
                                
                                # Store shifts in the displacement map
                                row_idx = r // step
                                col_idx = c // step
                                range_shifts[row_idx, col_idx] = shift[0]
                                azimuth_shifts[row_idx, col_idx] = shift[1]
                                
                                # Log extreme values to catch potential issues
                                if abs(shift[0]) > 5 or abs(shift[1]) > 5:
                                    self.log(f"Warning: Large displacement at ({row_idx}, {col_idx}): {shift}")
                                
                            except Exception as e:
                                error_count += 1
                                self.log(f"Error in phase_cross_correlation at position ({r}, {c}): {str(e)}")
                                if error_count < 10:  # Only print traceback for first few errors
                                    import traceback
                                    self.log(f"Traceback: {traceback.format_exc()}")
                                elif error_count == 10:
                                    self.log(f"Too many errors, suppressing detailed tracebacks...")
                                continue
                        except Exception as e:
                            # Outer exception handler for any other errors
                            self.log(f"Unexpected error in window processing at ({r}, {c}): {str(e)}")
                            import traceback
                            self.log(traceback.format_exc())
                
                self.log(f"Completed image pair {i+1} with {error_count} errors out of {total_windows} windows")
                
                # Log displacement statistics
                if np.size(range_shifts) > 0:
                    r_min, r_max = np.min(range_shifts), np.max(range_shifts)
                    a_min, a_max = np.min(azimuth_shifts), np.max(azimuth_shifts)
                    self.log(f"Range displacement range: {r_min:.4f} to {r_max:.4f}")
                    self.log(f"Azimuth displacement range: {a_min:.4f} to {a_max:.4f}")
                
                # Store displacement maps for this pair
                self.displacement_maps.append((range_shifts, azimuth_shifts))
                
            except Exception as e:
                import traceback
                self.log(f"Critical error in displacement estimation for pair {i+1}: {e}")
                self.log(traceback.format_exc())
                self.last_error = traceback.format_exc()
                return False
                
        self.log(f"Displacement estimation completed successfully for {len(self.displacement_maps)} image pairs")
        return True
    
    def analyze_time_series(self, measurement_points):
        """
        Analyze displacement time series at specified points and compute frequency spectra.

        Parameters:
        - measurement_points: list of tuples, (row, col) coordinates

        Returns:
        - bool, True if successful

        Notes:
        - Requires self.displacement_maps and self.subaperture_times to be set.
        """
        if not hasattr(self, 'displacement_maps') or not self.displacement_maps:
            print("Error: No displacement maps available")
            return False
        
        self.time_series = {'range': {}, 'azimuth': {}}
        self.frequency_spectra = {'range': {}, 'azimuth': {}}
        step = int(self.window_size * (1 - self.overlap))
        
        for idx, (row, col) in enumerate(measurement_points):
            map_row = row // step
            map_col = col // step
            range_series = []
            azimuth_series = []
            
            for range_map, azimuth_map in self.displacement_maps:
                if (0 <= map_row < range_map.shape[0] and 
                    0 <= map_col < range_map.shape[1]):
                    range_series.append(range_map[map_row, map_col])
                    azimuth_series.append(azimuth_map[map_row, map_col])
            
            self.time_series['range'][idx] = np.array(range_series)
            self.time_series['azimuth'][idx] = np.array(azimuth_series)
            
            # Calculate sampling frequency
            if len(self.subaperture_times) > 1:
                delta_t = np.mean(np.diff(self.subaperture_times))
                fs = 1 / delta_t if delta_t > 0 else 1.0
            else:
                fs = 1.0  # Fallback

            if len(range_series) > 1:
                range_fft = np.abs(fft(range_series))
                azimuth_fft = np.abs(fft(azimuth_series))
                n = len(range_series)
                freq = fftfreq(n, 1 / fs)[:n // 2]
                self.frequency_spectra['range'][idx] = (freq, range_fft[:n // 2])
                self.frequency_spectra['azimuth'][idx] = (freq, azimuth_fft[:n // 2])
        
        return True
    
    def calculate_vibration_energy_map(self):
        """
        Calculate vibration energy map from displacement maps
        
        Returns
        -------
        bool
            True if vibration energy map was calculated successfully, False otherwise
        """
        if self.displacement_maps is None or len(self.displacement_maps) == 0:
            print("Error: No displacement maps available")
            return False
        
        # Get dimensions of displacement maps
        range_map, azimuth_map = self.displacement_maps[0]
        rows, cols = range_map.shape
        
        # Initialize vibration energy map
        self.vibration_energy_map = np.zeros((rows, cols))
        
        # For each position in the displacement maps
        for r in range(rows):
            for c in range(cols):
                # Extract time series for range and azimuth
                range_series = []
                azimuth_series = []
                
                for range_map, azimuth_map in self.displacement_maps:
                    range_series.append(range_map[r, c])
                    azimuth_series.append(azimuth_map[r, c])
                
                # Calculate vibration energy as the sum of squared displacements
                range_energy = np.sum(np.square(range_series))
                azimuth_energy = np.sum(np.square(azimuth_series))
                
                # Total energy is the sum of range and azimuth energies
                total_energy = range_energy + azimuth_energy
                
                # Store in vibration energy map
                self.vibration_energy_map[r, c] = total_energy
        
        # Convert to dB scale
        # Add a small value to avoid log(0)
        min_non_zero = np.min(self.vibration_energy_map[self.vibration_energy_map > 0]) if np.any(self.vibration_energy_map > 0) else 1e-10
        self.vibration_energy_map[self.vibration_energy_map == 0] = min_non_zero / 10
        self.vibration_energy_map_db = 10 * np.log10(self.vibration_energy_map)
        
        # Normalize to 0 to -25 dB range as in the example image
        max_db = np.max(self.vibration_energy_map_db)
        self.vibration_energy_map_db = self.vibration_energy_map_db - max_db  # 0 is the maximum
        self.vibration_energy_map_db = np.maximum(self.vibration_energy_map_db, -25)  # Clip at -25 dB
        
        return True
    
    def detect_ship_regions(self, num_regions=3, energy_threshold=-15):
        """
        Detect ship regions based on vibration energy
        
        Parameters
        ----------
        num_regions : int, optional
            Number of regions to detect (default is 3)
        energy_threshold : float, optional
            Energy threshold in dB for ship detection (default is -15)
            
        Returns
        -------
        bool
            True if ship regions were detected successfully, False otherwise
        """
        if self.vibration_energy_map_db is None:
            print("Error: No vibration energy map available")
            return False
        
        # Threshold the energy map to find ship pixels
        ship_mask = self.vibration_energy_map_db > energy_threshold
        
        # If no ship pixels found, return False
        if not np.any(ship_mask):
            print("Error: No ship regions detected")
            return False
        
        # Find connected components (simplified approach)
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(ship_mask)
        
        # If fewer regions than requested, adjust num_regions
        num_regions = min(num_regions, num_features)
        
        # Find the largest regions
        region_sizes = np.zeros(num_features + 1)
        for i in range(1, num_features + 1):
            region_sizes[i] = np.sum(labeled_array == i)
        
        # Sort regions by size (descending)
        sorted_regions = np.argsort(-region_sizes[1:]) + 1
        
        # Select the largest regions
        selected_regions = sorted_regions[:num_regions]
        
        # Store ship regions
        self.ship_regions = []
        for i, region_id in enumerate(selected_regions):
            # Get region mask
            region_mask = labeled_array == region_id
            
            # Find region centroid
            r_indices, c_indices = np.where(region_mask)
            centroid_r = np.mean(r_indices)
            centroid_c = np.mean(c_indices)
            
            # Store region info
            self.ship_regions.append({
                'id': i + 1,  # 1-based indexing as in the example
                'mask': region_mask,
                'centroid': (centroid_r, centroid_c)
            })
        
        return True
    
    def plot_results(self, point_index, output_dir=None, output_file=None):
        """Plot results for a specific measurement point (utility function that can save to specified directory)"""
        if not hasattr(self, 'time_series') or not self.time_series:
            self.log("No time series data available to plot")
            return False
            
        if point_index >= len(self.time_series['range']):
            self.log(f"Point index {point_index} out of range")
            return False
            
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot range time series
        axs[0, 0].plot(self.time_series['range'][point_index])
        axs[0, 0].set_title(f'Range Displacement Time Series (Point {point_index})')
        axs[0, 0].set_xlabel('Time (samples)')
        axs[0, 0].set_ylabel('Displacement (pixels)')
        axs[0, 0].grid(True)
        
        # Plot azimuth time series
        axs[0, 1].plot(self.time_series['azimuth'][point_index])
        axs[0, 1].set_title(f'Azimuth Displacement Time Series (Point {point_index})')
        axs[0, 1].set_xlabel('Time (samples)')
        axs[0, 1].set_ylabel('Displacement (pixels)')
        axs[0, 1].grid(True)
        
        # Plot range frequency spectrum
        freq, spectrum = self.frequency_spectra['range'][point_index]
        axs[1, 0].plot(freq, spectrum)
        axs[1, 0].set_title(f'Range Frequency Spectrum (Point {point_index})')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Amplitude')
        axs[1, 0].grid(True)
        
        # Plot azimuth frequency spectrum
        freq, spectrum = self.frequency_spectra['azimuth'][point_index]
        axs[1, 1].plot(freq, spectrum)
        axs[1, 1].set_title(f'Azimuth Frequency Spectrum (Point {point_index})')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Amplitude')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save the plot if a directory or file is specified
        if output_file:
            plt.savefig(output_file)
            self.log(f"Saved point {point_index} results to {output_file}")
        elif output_dir:
            output_path = os.path.join(output_dir, f'point_{point_index}_results.png')
            plt.savefig(output_path)
            self.log(f"Saved point {point_index} results to {output_path}")
            
        plt.close(fig)
        return True
    
    def plot_vibration_energy_map(self, output_dir=None, output_file=None):
        """Plot the vibration energy map (utility function that can save to specified directory)"""
        if not hasattr(self, 'vibration_energy_map_db') or self.vibration_energy_map_db is None:
            self.log("No vibration energy map available to plot")
            return False
            
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the first SLC image for reference
        if self.slc_images is not None and len(self.slc_images) > 0:
            axs[0].imshow(np.abs(self.slc_images[0]), cmap='gray', aspect='auto')
            axs[0].set_title('SLC Image (First Frame)')
            axs[0].set_xlabel('Range (pixels)')
            axs[0].set_ylabel('Azimuth (pixels)')
        
        # Plot vibration energy map
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=-25, vmax=0)
        im = axs[1].imshow(self.vibration_energy_map_db, cmap=cmap, norm=norm, aspect='auto')
        axs[1].set_title('Vibration Energy Map (dB)')
        axs[1].set_xlabel('Range (pixels)')
        axs[1].set_ylabel('Azimuth (pixels)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axs[1])
        cbar.set_label('Vibration Energy (dB)')
        
        plt.tight_layout()
        
        # Save the plot if a directory or file is specified
        if output_file:
            plt.savefig(output_file)
            self.log(f"Saved vibration energy map to {output_file}")
        elif output_dir:
            output_path = os.path.join(output_dir, 'vibration_energy_map.png')
            plt.savefig(output_path)
            self.log(f"Saved vibration energy map to {output_path}")
            
        plt.close(fig)
        return True
    
    def identify_dominant_frequencies(self, measurement_point_idx, threshold=0.5):
        """
        Identify dominant frequencies in the spectrum for a specific measurement point
        
        Parameters
        ----------
        measurement_point_idx : int
            Index of the measurement point to analyze
        threshold : float, optional
            Threshold for peak detection (default is 0.5)
            
        Returns
        -------
        dict
            Dictionary containing dominant frequencies for range and azimuth
        """
        if (self.frequency_spectra is None or 
            measurement_point_idx not in self.frequency_spectra['range'] or 
            measurement_point_idx not in self.frequency_spectra['azimuth']):
            print(f"Error: No frequency spectra available for measurement point {measurement_point_idx}")
            return None
        
        results = {'range': [], 'azimuth': []}
        
        # Analyze range spectrum
        freq, spectrum = self.frequency_spectra['range'][measurement_point_idx]
        # Normalize spectrum - fix for division by zero
        max_val = np.max(spectrum)
        if max_val > 0:
            norm_spectrum = spectrum / max_val
            # Find peaks
            peaks, _ = signal.find_peaks(norm_spectrum, height=threshold)
            # Get frequencies and amplitudes of peaks
            peak_freqs = freq[peaks]
            peak_amps = norm_spectrum[peaks]
            # Sort by amplitude
            sorted_idx = np.argsort(peak_amps)[::-1]
            results['range'] = [(peak_freqs[i], peak_amps[i]) for i in sorted_idx]
        
        # Analyze azimuth spectrum
        freq, spectrum = self.frequency_spectra['azimuth'][measurement_point_idx]
        # Normalize spectrum - fix for division by zero
        max_val = np.max(spectrum)
        if max_val > 0:
            norm_spectrum = spectrum / max_val
            # Find peaks
            peaks, _ = signal.find_peaks(norm_spectrum, height=threshold)
            # Get frequencies and amplitudes of peaks
            peak_freqs = freq[peaks]
            peak_amps = norm_spectrum[peaks]
            # Sort by amplitude
            sorted_idx = np.argsort(peak_amps)[::-1]
            results['azimuth'] = [(peak_freqs[i], peak_amps[i]) for i in sorted_idx]
        
        return results

    def create_subapertures(self):
        """Create subapertures from the raw data if not already created"""
        self.log("Starting subaperture creation...")
        
        # If subapertures already exist, just return success
        if hasattr(self, 'subapertures') and self.subapertures and len(self.subapertures) > 0:
            self.log(f"Subapertures already exist ({len(self.subapertures)} found), reusing them")
            return True
            
        # Otherwise, check for raw data and create subapertures
        if not hasattr(self, 'raw_data') or self.raw_data is None:
            self.log("Error: No raw data available")
            return False
            
        try:
            # Validate raw data
            self.log(f"Raw data shape: {self.raw_data.shape}")
            if np.isnan(self.raw_data).any():
                self.log("Warning: Raw data contains NaN values")
            if np.isinf(self.raw_data).any():
                self.log("Warning: Raw data contains Inf values")
            
            # Log descriptive statistics of raw data
            data_mag = np.abs(self.raw_data)
            self.log(f"Raw data magnitude range: {np.min(data_mag):.4f} to {np.max(data_mag):.4f}")
            self.log(f"Raw data magnitude mean: {np.mean(data_mag):.4f}, std: {np.std(data_mag):.4f}")
            
            # Call the class method to split aperture
            self.log(f"Creating {self.num_subapertures} subapertures with overlap {self.overlap}...")
            self.subapertures, self.subaperture_times = self._split_aperture(
                self.raw_data, self.num_subapertures, self.overlap
            )
            self.log(f"Created {len(self.subapertures)} sub-apertures")
            
            # Validate created subapertures
            for i, subap in enumerate(self.subapertures):
                self.log(f"Subaperture {i+1} shape: {subap.shape}")
                if np.all(np.abs(subap) < 1e-10):
                    self.log(f"Warning: Subaperture {i+1} appears to be empty (all zeros)")
            
            # Log timing information
            self.log(f"Subaperture timing: {self.subaperture_times}")
            
            return True
        except Exception as e:
            self.log(f"Error creating subapertures: {str(e)}")
            import traceback
            trace = traceback.format_exc()
            self.log(trace)
            self.last_error = f"{str(e)}\n\n{trace}"
            return False

    def focus_subapertures(self):
        """Focus the subapertures to create SLC images"""
        self.log("Starting subaperture focusing...")
        
        # If SLC images already exist, just return success
        if hasattr(self, 'slc_images') and self.slc_images and len(self.slc_images) > 0:
            self.log(f"SLC images already exist ({len(self.slc_images)} found), reusing them")
            return True
            
        # Otherwise check for subapertures
        if not hasattr(self, 'subapertures') or not self.subapertures or len(self.subapertures) == 0:
            self.log("Error: No subapertures available to focus")
            return False
            
        try:
            # Focus the subapertures
            self.log(f"Focusing {len(self.subapertures)} subapertures...")
            
            # In this initial implementation, we use magnitude as the SLC image
            # In a more advanced implementation, this would involve proper focusing
            self.slc_images = []
            for i, subap in enumerate(self.subapertures):
                self.log(f"Focusing subaperture {i+1}/{len(self.subapertures)}...")
                
                # Check if subap contains NaN or Inf values
                if np.isnan(subap).any():
                    self.log(f"Warning: Subaperture {i+1} contains NaN values")
                if np.isinf(subap).any():
                    self.log(f"Warning: Subaperture {i+1} contains Inf values")
                
                # Simply use the magnitude as the SLC image for now
                # This is a placeholder for proper focusing algorithms
                slc = np.abs(subap)
                
                # Check if focusing produced valid results
                if np.all(slc < 1e-10):
                    self.log(f"Warning: SLC image {i+1} appears to be empty (all values close to zero)")
                
                # Log SLC image statistics
                self.log(f"SLC image {i+1} value range: {np.min(slc):.4f} to {np.max(slc):.4f}")
                self.log(f"SLC image {i+1} mean: {np.mean(slc):.4f}, std: {np.std(slc):.4f}")
                
                self.slc_images.append(slc)
                
            self.log(f"Created {len(self.slc_images)} SLC images")
            
            # Store normalized versions for visualization
            if self.debug_mode:
                self.normalized_slc_images = []
                for i, slc in enumerate(self.slc_images):
                    # Normalize for better visualization
                    norm_slc = slc / np.percentile(slc, 99) if np.percentile(slc, 99) > 0 else slc
                    self.normalized_slc_images.append(norm_slc)
                self.log("Created normalized SLC images for visualization")
                
            return True
        except Exception as e:
            self.log(f"Error focusing subapertures: {str(e)}")
            import traceback
            trace = traceback.format_exc()
            self.log(trace)
            self.last_error = f"{str(e)}\n\n{trace}"
            return False

    def estimate_displacement_memory_efficient(self):
        """
        Memory-efficient version of displacement estimation that processes data in chunks
        to avoid memory exhaustion
        
        Returns
        -------
        bool
            True if displacement was estimated successfully, False otherwise
        """
        self.log("Starting memory-efficient displacement estimation...")
        
        if self.slc_images is None or len(self.slc_images) < 2:
            self.log("Error: No SLC images available for displacement estimation")
            return False
        
        # Initialize displacement maps
        self.log(f"Initializing displacement maps for {len(self.slc_images)-1} image pairs")
        self.displacement_maps = []
        
        # Calculate step size based on window size and overlap
        step = int(self.window_size * (1 - self.overlap))
        self.log(f"Using window size: {self.window_size}, overlap: {self.overlap}, step size: {step}")
        
        # Process data in chunks to reduce memory usage
        CHUNK_SIZE = 10  # Number of rows to process at once
        
        # For each pair of adjacent sub-apertures
        for i in range(len(self.slc_images) - 1):
            self.log(f"Processing image pair {i+1}/{len(self.slc_images)-1}...")
            
            ref_image = self.slc_images[i]
            sec_image = self.slc_images[i + 1]
            
            # Get image dimensions
            rows, cols = ref_image.shape
            self.log(f"Image dimensions: {rows}x{cols}")
            
            # Initialize displacement map for this pair
            range_shifts = np.zeros((rows // step, cols // step))
            azimuth_shifts = np.zeros((rows // step, cols // step))
            self.log(f"Displacement map dimensions: {range_shifts.shape}")
            
            # Check for valid image data
            if np.isnan(ref_image).any() or np.isinf(ref_image).any():
                self.log(f"Warning: Reference image contains NaN or Inf values")
            if np.isnan(sec_image).any() or np.isinf(sec_image).any():
                self.log(f"Warning: Secondary image contains NaN or Inf values")
                
            # Calculate total number of windows for progress tracking
            num_rows_windows = (rows - self.window_size) // step + 1
            num_cols_windows = (cols - self.window_size) // step + 1
            total_windows = num_rows_windows * num_cols_windows
            self.log(f"Total windows to process: {total_windows}")
            
            # Track errors and processed windows
            error_count = 0
            processed_windows = 0
            
            # Process in chunks of rows
            for chunk_start in range(0, rows - self.window_size, CHUNK_SIZE * step):
                chunk_end = min(chunk_start + CHUNK_SIZE * step, rows - self.window_size)
                self.log(f"Processing chunk from row {chunk_start} to {chunk_end} ({chunk_end-chunk_start} rows)")
                
                # Process each window position in this chunk
                for r in range(chunk_start, chunk_end, step):
                    for c in range(0, cols - self.window_size, step):
                        processed_windows += 1
                        
                        # Log progress periodically
                        if processed_windows % 100 == 0 or processed_windows == total_windows:
                            progress_pct = (processed_windows / total_windows) * 100
                            self.log(f"Processed {processed_windows}/{total_windows} windows ({progress_pct:.1f}%)")
                        
                        # Extract windows from both images
                        ref_window = ref_image[r:r+self.window_size, c:c+self.window_size]
                        sec_window = sec_image[r:r+self.window_size, c:c+self.window_size]
                        
                        # Check window validity
                        if np.all(ref_window == 0) or np.all(sec_window == 0):
                            continue
                        
                        # Check window contrast
                        ref_std = np.std(ref_window)
                        sec_std = np.std(sec_window)
                        if ref_std < 1e-3 or sec_std < 1e-3:
                            continue
                        
                        # Check for NaN or Inf values
                        if np.isnan(ref_window).any() or np.isnan(sec_window).any() or \
                           np.isinf(ref_window).any() or np.isinf(sec_window).any():
                            continue
                        
                        # Calculate sub-pixel shift
                        try:
                            shift, error, diffphase = phase_cross_correlation(
                                ref_window, sec_window, upsample_factor=100
                            )
                            
                            # Store shifts in displacement maps
                            row_idx = r // step
                            col_idx = c // step
                            range_shifts[row_idx, col_idx] = shift[0]
                            azimuth_shifts[row_idx, col_idx] = shift[1]
                            
                        except Exception as e:
                            error_count += 1
                            if error_count < 10:
                                self.log(f"Error in window at ({r}, {c}): {str(e)}")
                            elif error_count == 10:
                                self.log("Suppressing further error messages...")
                
                # Explicitly free memory after processing each chunk
                import gc
                gc.collect()
                
                # Save intermediate results after each chunk
                if self.debug_mode:
                    self.log(f"Saving intermediate results after chunk {chunk_start}-{chunk_end}")
                    # Could implement intermediate saving here
            
            self.log(f"Completed image pair {i+1} with {error_count} errors out of {total_windows} windows")
            
            # Store displacement maps for this pair
            self.displacement_maps.append((range_shifts, azimuth_shifts))
            
            # Log displacement statistics
            if np.size(range_shifts) > 0:
                r_min, r_max = np.min(range_shifts), np.max(range_shifts)
                a_min, a_max = np.min(azimuth_shifts), np.max(azimuth_shifts)
                self.log(f"Range displacement range: {r_min:.4f} to {r_max:.4f}")
                self.log(f"Azimuth displacement range: {a_min:.4f} to {a_max:.4f}")
        
        self.log(f"Memory-efficient displacement estimation completed successfully")
        return True
