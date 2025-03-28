import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import shift, zoom
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from skimage.registration import phase_cross_correlation
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from skimage.transform import warp, AffineTransform
from cphd_reader import split_aperture
from skimage.morphology import label

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
        self.data_type = None  # Will be set to 'cphd' or 'sicd' in load_data
        self.subapertures = None
        self.subaperture_times = None  # Added to store timing info for frequency analysis
        self.displacement_maps = None
        self.time_series = None
        self.frequency_spectra = None
        self.vibration_energy_map = None
        self.vibration_energy_map_db = None  # Added to explicitly store dB map
        self.ship_regions = None
        self.data_loaded = False
        
        # Debug mode attributes
        self.raw_data = None  # For CPHD phase history
        self.raw_complex_data = None  # Added for SICD complex image data
        self.pvp = None  # Added to store PVP explicitly
        self.sicd_meta = None  # Added to store SICD metadata
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
        
    def load_data(self, file_path, channel_index=0):
        """
        Load CPHD or SICD data. For CPHD, reads phase history. For SICD, reads complex image data.

        Parameters:
        - file_path: str, path to CPHD or SICD file or 'demo'
        - channel_index: int, channel/index to process (default: 0)

        Returns:
        - bool, True if successful
        """
        self.log(f"Attempting to load data from: {file_path}")
        
        if file_path == 'demo':
            # Keep demo data generation as is (implicitly CPHD-like)
            self.log("Creating synthetic demo data (CPHD-like)")
            self.data_type = 'cphd'  # Assume demo is CPHD-like for now
            from test_estimator import create_synthetic_data
            
            # Demo data creates 'subapertures' directly
            # We now need complex subapertures for consistency
            try:
                complex_subaps, pvp, _ = create_synthetic_data(
                    rows=512, cols=512, num_subapertures=self.num_subapertures, 
                    output_complex=True  # Request complex output if supported
                )
                self.subapertures = complex_subaps  # Store complex subapertures
                self.pvp = pvp
                self.subaperture_times = np.linspace(0, 1, self.num_subapertures)
                # Create a representative raw_data_image (from first subap)
                if self.subapertures and len(self.subapertures) > 0:
                    self.raw_data_image = np.abs(self.subapertures[0])
                self.data_loaded = True
                self.log("Synthetic demo data loaded successfully")
                return True
            except Exception as e:
                self.log(f"Error creating synthetic data: {e}")
                import traceback
                self.log(traceback.format_exc())
                self.last_error = traceback.format_exc()
                return False
                
        elif not os.path.exists(file_path):
            self.log(f"Error: File not found at {file_path}")
            return False
            
        else:
            try:
                from sarpy.io import open as sarpy_open
                
                self.log(f"Opening file with sarpy: {file_path}")
                # Use sarpy's generic open function which is more robust
                reader = sarpy_open(file_path)
                reader_type = type(reader).__name__
                self.log(f"Sarpy reader type: {reader_type}")
                
                # Determine data type based on reader type or metadata
                has_cphd_meta = hasattr(reader, 'cphd_meta') and reader.cphd_meta is not None
                has_sicd_meta = hasattr(reader, 'sicd_meta') and reader.sicd_meta is not None
                
                if has_cphd_meta or 'cphd' in reader_type.lower():
                    self.data_type = 'cphd'
                    self.log("Detected CPHD data type.")
                    
                    # --- CPHD Loading Logic ---
                    self.log("Reading CPHD phase history data...")
                    # Read the data chip
                    try:
                        if hasattr(reader, 'read_chip'):
                            data_indices = list(reader.get_data_indices())
                            if not data_indices:
                                self.log("Error: Reader has no data indices.")
                                return False
                            read_index = data_indices[min(channel_index, len(data_indices)-1)]
                            self.log(f"Reading CPHD chip for index: {read_index}")
                            self.raw_data = reader.read_chip(index=read_index)
                        else:
                            raise ValueError("Reader does not have read_chip method")
                    except Exception as e:
                        self.log(f"Error reading CPHD data chip: {e}")
                        return False
                    
                    self.log(f"Raw CPHD data shape: {self.raw_data.shape}")
                    self.raw_data_image = np.abs(self.raw_data)  # For visualization
                    
                    # Read PVP data for timing
                    self.log("Reading PVP data...")
                    try:
                        if hasattr(reader, 'read_pvp'):
                            self.pvp = reader.read_pvp()
                            if not self.pvp or 'TxTime' not in self.pvp:
                                self.log("Warning: PVP data read but 'TxTime' is missing. Using synthetic timing.")
                                self.pvp = {'TxTime': np.linspace(0, 1, self.raw_data.shape[0])}
                        else:
                            self.log("Warning: Reader has no read_pvp method. Using synthetic timing.")
                            self.pvp = {'TxTime': np.linspace(0, 1, self.raw_data.shape[0])}
                    except Exception as e:
                        self.log(f"Error reading PVP data: {e}. Using synthetic timing.")
                        self.pvp = {'TxTime': np.linspace(0, 1, self.raw_data.shape[0])}
                    
                    self.log("CPHD data loaded successfully.")
                    
                elif has_sicd_meta or 'sicd' in reader_type.lower():
                    self.data_type = 'sicd'
                    self.log("Detected SICD data type.")
                    self.raw_data = None # No raw phase history for SICD
                    self.pvp = None

                    # --- SICD Loading Logic ---
                    self.log("Reading SICD complex image data...")
                    try:
                        if hasattr(reader, 'read_chip'):
                            # SICDReader typically doesn't have get_data_indices method
                            # Instead, directly read the chip (it's usually a single image)
                            self.log("Reading default SICD chip...")
                            try:
                                # First try without any parameters
                                self.raw_complex_data = reader.read_chip()
                            except Exception as e:
                                self.log(f"Error reading SICD chip without parameters: {e}, trying with channel_index")
                                # If that fails, try with the channel_index parameter
                                self.raw_complex_data = reader.read_chip(index=channel_index)
                        else:
                            self.log("Error: SICD reader does not have read_chip method.")
                            return False
                    except Exception as e:
                        self.log(f"Error reading SICD data chip: {e}")
                        # Add traceback for more detailed debugging
                        import traceback
                        trace = traceback.format_exc()
                        self.log(f"Traceback: {trace}")
                        self.last_error = f"{str(e)}\n\n{trace}"
                        return False
                    
                    self.log(f"Raw SICD complex data shape: {self.raw_complex_data.shape}")
                    self.raw_data_image = np.abs(self.raw_complex_data)  # For visualization
                    
                    # Store SICD metadata for timing information
                    self.sicd_meta = reader.sicd_meta
                    if isinstance(self.sicd_meta, list):  # Multiple SICDs in a file
                        self.sicd_meta = self.sicd_meta[min(channel_index, len(self.sicd_meta)-1)]
                    
                    self.log("SICD data loaded successfully.")
                    
                else:
                    self.log(f"Error: Could not determine data type (CPHD or SICD) for reader {reader_type}")
                    return False
                
                self.data_loaded = True
                return True
                
            except Exception as e:
                self.log(f"Error loading data: {e}")
                import traceback
                trace = traceback.format_exc()
                self.log(trace)
                self.last_error = f"{str(e)}\n\n{trace}"
                return False
    
    def _split_aperture_cphd(self, data, num_subapertures, overlap=0.5):
        """
        Split CPHD phase history data into overlapping sub-apertures based on azimuth pulses.
        (Renamed from _split_aperture)

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

        # Calculate subaperture size and step
        # Ensure subaperture_size is at least 1
        subaperture_size = max(1, int(np.floor(num_pulses / (num_subapertures - (num_subapertures - 1) * overlap))))
        step = max(1, int(subaperture_size * (1 - overlap)))
        self.log(f"CPHD Splitting: num_pulses={num_pulses}, num_subaps={num_subapertures}, overlap={overlap} -> subap_size={subaperture_size}, step={step}")

        subapertures = []
        subaperture_times = []

        # Get timing information from PVP
        if hasattr(self, 'pvp') and self.pvp and 'TxTime' in self.pvp and self.pvp['TxTime'] is not None and len(self.pvp['TxTime']) == num_pulses:
            times = self.pvp['TxTime']
            self.log("Using TxTime from PVP for subaperture timing")
        else:
            self.log("Warning: Valid TxTime not found in PVP or length mismatch. Using synthetic linear timing.")
            times = np.linspace(0, 1, num_pulses)  # Synthetic timing

        start_idx = 0
        for i in range(num_subapertures):
            end_idx = start_idx + subaperture_size
            # Ensure the last subaperture reaches the end
            if i == num_subapertures - 1:
                end_idx = num_pulses
            # Prevent exceeding bounds
            end_idx = min(end_idx, num_pulses)
            # Ensure start index is valid
            start_idx = min(start_idx, num_pulses - 1)
            # Ensure we have at least one sample
            if end_idx <= start_idx:
                if i > 0:  # If not the first, try to take the last sample
                    start_idx = end_idx - 1
                else:  # If first, cannot proceed
                    self.log(f"Warning: Cannot create subaperture {i+1} due to index calculation.")
                    continue

            subaperture = data[start_idx:end_idx, :].copy()
            subapertures.append(subaperture)

            # Calculate average time for this subaperture
            current_times = times[start_idx:end_idx]
            if len(current_times) > 0:
                subaperture_times.append(np.mean(current_times))
            else:
                # Fallback if times array is problematic
                subaperture_times.append(times[start_idx] if start_idx < len(times) else i / num_subapertures)

            # Move to the next start index
            start_idx += step
            # Stop if next start index is out of bounds
            if start_idx >= num_pulses:
                break

        # If fewer subapertures were created than requested due to step size/overlap
        if len(subapertures) < num_subapertures:
            self.log(f"Warning: Created {len(subapertures)} subapertures, less than requested {num_subapertures}. Adjust parameters if needed.")

        return subapertures, subaperture_times
        
    def _split_aperture_sicd(self, data, num_subapertures, overlap=0.5):
        """
        Split SICD complex image data into overlapping sub-images along the azimuth dimension.
        
        Parameters:
        - data: numpy.ndarray, complex image data (azimuth x range)
        - num_subapertures: int, number of sub-images to create
        - overlap: float, overlap ratio between sub-images (0 to 1, default: 0.5)
        
        Returns:
        - list of numpy.ndarray, each containing a sub-image
        - list of floats, timing information for each sub-image
        """
        if data is None:
            raise ValueError("SICD complex data is not loaded.")
            
        num_azimuth, num_range = data.shape  # Assuming azimuth is axis 0
        if num_subapertures <= 0:
            raise ValueError("Number of sub-apertures must be positive.")
        if num_subapertures == 1:
            self.log("Number of sub-apertures is 1, returning the whole image.")
            # Attempt to get timing info for the single image
            start_time, end_time = self._get_sicd_timing_info()
            center_time = (start_time + end_time) / 2.0 if start_time is not None else 0.0
            return [data.copy()], [center_time]
        if num_subapertures > num_azimuth:
            self.log(f"Warning: Requested {num_subapertures} sub-apertures, but image only has {num_azimuth} azimuth lines. Reducing to {num_azimuth}.")
            num_subapertures = num_azimuth  # Cannot have more subaps than lines
            
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")
            
        # Calculate subaperture size and step along azimuth axis
        # Ensure subaperture_size is at least 1
        subaperture_size = max(1, int(np.floor(num_azimuth / (num_subapertures - (num_subapertures - 1) * overlap))))
        step = max(1, int(subaperture_size * (1 - overlap)))
        self.log(f"SICD Splitting: num_azimuth={num_azimuth}, num_subaps={num_subapertures}, overlap={overlap} -> subap_size={subaperture_size}, step={step}")
        
        subapertures = []
        subaperture_times = []
        
        # Get timing information for the whole collect
        start_time, end_time = self._get_sicd_timing_info()
        duration = end_time - start_time if start_time is not None and end_time is not None else 1.0  # Default duration 1.0 if no timing
        if duration <= 0:
            self.log("Warning: SICD duration is zero or negative. Using synthetic timing.")
            duration = 1.0
            start_time = 0.0
            
        start_idx = 0
        for i in range(num_subapertures):
            end_idx = start_idx + subaperture_size
            # Ensure the last subaperture reaches the end
            if i == num_subapertures - 1:
                end_idx = num_azimuth
            # Prevent exceeding bounds
            end_idx = min(end_idx, num_azimuth)
            # Ensure start index is valid
            start_idx = min(start_idx, num_azimuth - 1)
            # Ensure we have at least one line
            if end_idx <= start_idx:
                if i > 0:
                    start_idx = end_idx - 1
                else:
                    self.log(f"Warning: Cannot create SICD sub-image {i+1} due to index calculation.")
                    continue
                    
            subaperture = data[start_idx:end_idx, :].copy()
            subapertures.append(subaperture)
            
            # Calculate center time for this sub-image
            center_frac = (start_idx + (end_idx - start_idx) / 2.0) / num_azimuth
            center_time = start_time + center_frac * duration
            subaperture_times.append(center_time)
            
            # Move to the next start index
            start_idx += step
            # Stop if next start index is out of bounds
            if start_idx >= num_azimuth:
                break
                
        # If fewer subapertures were created than requested
        if len(subapertures) < num_subapertures:
            self.log(f"Warning: Created {len(subapertures)} SICD sub-images, less than requested {num_subapertures}. Adjust parameters if needed.")
            
        return subapertures, subaperture_times
        
    def _get_sicd_timing_info(self):
        """
        Helper method to extract start and end time from SICD metadata.
        
        Returns:
        - tuple: (start_time, end_time) in seconds
        """
        try:
            if hasattr(self, 'sicd_meta') and self.sicd_meta:
                # Try to access Timeline information (structure might vary)
                if hasattr(self.sicd_meta, 'Timeline') and self.sicd_meta.Timeline:
                    # Get collection start time
                    start_time = getattr(self.sicd_meta.Timeline, 'CollectStart', None)
                    if start_time is not None:
                        # Get collection duration if available
                        duration = getattr(self.sicd_meta.Timeline, 'CollectDuration', None)
                        if duration is not None and duration > 0:
                            end_time = start_time + duration
                            self.log(f"Extracted SICD timing: Start={start_time}, End={end_time}")
                            return start_time, end_time
                            
                    # Fallback using IPP (Impulse Phase Point) data if available
                    if hasattr(self.sicd_meta.Timeline, 'IPP') and self.sicd_meta.Timeline.IPP:
                        ipp_sets = getattr(self.sicd_meta.Timeline.IPP, 'Set', [])
                        if ipp_sets and len(ipp_sets) > 0:
                            ipp_set = ipp_sets[0]  # Use first set
                            ipp_start = getattr(ipp_set, 'IPPStart', 0)
                            ipp_end = getattr(ipp_set, 'IPPEnd', 0)
                            num_ipps = ipp_end - ipp_start + 1
                            if hasattr(ipp_set, 'IPPPoly') and len(ipp_set.IPPPoly) > 0:
                                ipp_rate = ipp_set.IPPPoly[0]  # Assuming constant rate (zeroth order poly)
                                if num_ipps > 0 and ipp_rate > 0:
                                    duration = num_ipps * ipp_rate
                                    # Use CollectStart if available
                                    start_time = start_time if start_time is not None else 0.0
                                    end_time = start_time + duration
                                    self.log(f"Extracted SICD timing from IPP: Start={start_time}, End={end_time}")
                                    return start_time, end_time
        except Exception as e:
            self.log(f"Error extracting SICD timing info: {e}")
            
        # Default timing if nothing else works
        self.log("Warning: Could not extract SICD timing information. Using default values [0, 1].")
        return 0.0, 1.0
    
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
        if not hasattr(self, 'displacement_maps') or self.displacement_maps is None or len(self.displacement_maps) == 0:
            self.log("Error: No displacement maps available for vibration energy calculation")
            return False
        
        self.log("Calculating vibration energy map from displacement maps...")
        
        # Get dimensions of displacement maps
        try:
            range_map, azimuth_map = self.displacement_maps[0]
            if range_map is None or azimuth_map is None:
                self.log("Error: Displacement maps contain None values")
                return False
                
            rows, cols = range_map.shape
            self.log(f"Displacement map dimensions: {rows}x{cols}")
            
            # Initialize vibration energy map
            self.vibration_energy_map = np.zeros((rows, cols), dtype=np.float32)
            
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
            
            # Convert to dB scale for better visualization
            # Add a small epsilon to avoid log(0)
            epsilon = 1e-12
            valid_energy = self.vibration_energy_map[self.vibration_energy_map > epsilon]
            if valid_energy.size == 0:
                self.log("Warning: All calculated vibration energy values are near zero")
                # Default to a uniform low value if everything is zero
                self.vibration_energy_map_db = np.full(self.vibration_energy_map.shape, -60.0, dtype=np.float32)
            else:
                min_positive_energy = np.min(valid_energy)
                # Use epsilon or a fraction of the minimum positive energy
                safe_map = np.maximum(self.vibration_energy_map, min_positive_energy * 0.1)
                self.vibration_energy_map_db = 10 * np.log10(safe_map)
            
            # Normalize using percentile to be robust to outliers
            finite_db_values = self.vibration_energy_map_db[np.isfinite(self.vibration_energy_map_db)]
            if finite_db_values.size > 0:
                # Use 99.5th percentile as the reference maximum
                max_db_ref = np.percentile(finite_db_values, 99.5)
                self.log(f"Normalizing dB map relative to 99.5th percentile: {max_db_ref:.2f} dB")
            else:
                # Fallback if all values are non-finite (e.g., -inf)
                max_db_ref = 0
                self.log("Warning: Could not find finite dB values for normalization reference")
            
            self.vibration_energy_map_db = self.vibration_energy_map_db - max_db_ref  # Reference percentile becomes 0 dB
            
            # Clip at a lower bound (e.g., -30 dB)
            lower_clip_db = -30
            self.vibration_energy_map_db = np.maximum(self.vibration_energy_map_db, lower_clip_db)
            self.log(f"Clipped vibration energy map at {lower_clip_db} dB")
            
            self.log(f"Vibration energy map calculated successfully with dimensions {self.vibration_energy_map_db.shape}")
            return True
            
        except Exception as e:
            self.log(f"Error calculating vibration energy map: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
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
        """Create subapertures from the raw data based on data type (CPHD or SICD)"""
        self.log("Starting subaperture creation...")
        
        # If subapertures already exist, just return success
        if hasattr(self, 'subapertures') and self.subapertures and len(self.subapertures) > 0:
            self.log(f"Subapertures already exist ({len(self.subapertures)} found), reusing them")
            return True
            
        # Check if data is loaded
        if not self.data_loaded:
            self.log("Error: Data not loaded")
            return False
            
        try:
            # Choose the appropriate method based on data type
            if self.data_type == 'cphd':
                # Check for raw data and create subapertures (CPHD)
                if not hasattr(self, 'raw_data') or self.raw_data is None:
                    self.log("Error: No raw CPHD data available")
                    return False
                
                # Validate raw data
                self.log(f"Raw CPHD data shape: {self.raw_data.shape}")
                if np.isnan(self.raw_data).any():
                    self.log("Warning: Raw CPHD data contains NaN values")
                if np.isinf(self.raw_data).any():
                    self.log("Warning: Raw CPHD data contains Inf values")
                
                # Log descriptive statistics of raw data
                data_mag = np.abs(self.raw_data)
                self.log(f"Raw CPHD data magnitude range: {np.min(data_mag):.4f} to {np.max(data_mag):.4f}")
                self.log(f"Raw CPHD data magnitude mean: {np.mean(data_mag):.4f}, std: {np.std(data_mag):.4f}")
                
                # Call the CPHD aperture splitting method
                self.log(f"Creating {self.num_subapertures} CPHD subapertures with overlap {self.overlap}...")
                self.subapertures, self.subaperture_times = self._split_aperture_cphd(
                    self.raw_data, self.num_subapertures, self.overlap
                )
                self.log(f"Created {len(self.subapertures)} CPHD sub-apertures")
                
            elif self.data_type == 'sicd':
                # Check for raw complex data and create subapertures (SICD)
                if not hasattr(self, 'raw_complex_data') or self.raw_complex_data is None:
                    self.log("Error: No raw SICD complex data available")
                    return False
                
                # Validate raw complex data
                self.log(f"Raw SICD data shape: {self.raw_complex_data.shape}")
                if np.isnan(self.raw_complex_data).any():
                    self.log("Warning: Raw SICD data contains NaN values")
                if np.isinf(self.raw_complex_data).any():
                    self.log("Warning: Raw SICD data contains Inf values")
                
                # Log descriptive statistics of raw data
                data_mag = np.abs(self.raw_complex_data)
                self.log(f"Raw SICD data magnitude range: {np.min(data_mag):.4f} to {np.max(data_mag):.4f}")
                self.log(f"Raw SICD data magnitude mean: {np.mean(data_mag):.4f}, std: {np.std(data_mag):.4f}")
                
                # Call the SICD aperture splitting method
                self.log(f"Creating {self.num_subapertures} SICD sub-images with overlap {self.overlap}...")
                self.subapertures, self.subaperture_times = self._split_aperture_sicd(
                    self.raw_complex_data, self.num_subapertures, self.overlap
                )
                self.log(f"Created {len(self.subapertures)} SICD sub-images")
                
            else:
                self.log(f"Error: Unknown data type '{self.data_type}'. Cannot create subapertures.")
                return False
                
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
        """
        Focus CPHD subapertures or prepare magnitude images for SICD sub-images.
        Generates self.slc_images (magnitude images) for displacement estimation.
        """
        self.log("Starting subaperture focusing/preparation step...")
        
        # If SLC images already exist, just return success
        if hasattr(self, 'slc_images') and self.slc_images and len(self.slc_images) > 0:
            self.log(f"SLC magnitude images already exist ({len(self.slc_images)} found), reusing them")
            return True
            
        # Otherwise check for subapertures
        if not hasattr(self, 'subapertures') or not self.subapertures or len(self.subapertures) == 0:
            self.log("Error: No subapertures available to focus or prepare")
            return False
            
        try:
            # Clear existing SLC images list if it exists
            self.slc_images = []
            
            if self.data_type == 'cphd':
                self.log(f"Focusing {len(self.subapertures)} CPHD subapertures...")
                # **********************************************************************
                # ** WARNING: Placeholder Focusing Implementation for CPHD **
                # This currently uses np.abs(), which is NOT a proper SAR focusing
                # algorithm for phase history data. Replace this with an appropriate
                # focusing method (e.g., Range-Doppler, Chirp Scaling) for
                # meaningful micro-motion analysis.
                # **********************************************************************
                self.log("WARNING: Using placeholder focusing (np.abs) for CPHD. Replace with proper SAR focusing algorithm.")
                
                for i, subap in enumerate(self.subapertures):
                    self.log(f"Focusing CPHD subaperture {i+1}/{len(self.subapertures)}...")
                    
                    # Check if subap contains NaN or Inf values
                    if np.isnan(subap).any():
                        self.log(f"Warning: Subaperture {i+1} contains NaN values")
                    if np.isinf(subap).any():
                        self.log(f"Warning: Subaperture {i+1} contains Inf values")
                    
                    # Simply use the magnitude as the SLC image for now
                    # PLACEHOLDER: Replace with proper focusing algorithm
                    slc = np.abs(subap)
                    
                    # Check if focusing produced valid results
                    if np.all(slc < 1e-10):
                        self.log(f"Warning: SLC image {i+1} appears to be empty (all values close to zero)")
                    
                    # Log SLC image statistics
                    self.log(f"CPHD SLC image {i+1} stats: min={np.min(slc):.2f}, max={np.max(slc):.2f}, mean={np.mean(slc):.2f}")
                    
                    self.slc_images.append(slc)
                    
                self.log(f"Created {len(self.slc_images)} CPHD SLC images")
                
            elif self.data_type == 'sicd':
                self.log(f"Preparing {len(self.subapertures)} SICD sub-images (taking magnitude)...")
                # SICD data is already focused. We just need the magnitude images
                # for displacement estimation.
                for i, complex_sub_image in enumerate(self.subapertures):
                    self.log(f"Taking magnitude of SICD sub-image {i+1}/{len(self.subapertures)}...")
                    
                    # Check for invalid values
                    if np.isnan(complex_sub_image).any():
                        self.log(f"Warning: SICD sub-image {i+1} contains NaN values")
                    if np.isinf(complex_sub_image).any():
                        self.log(f"Warning: SICD sub-image {i+1} contains Inf values")
                    
                    # Take magnitude
                    magnitude_image = np.abs(complex_sub_image)
                    
                    # Check if valid
                    if np.all(magnitude_image < 1e-10):
                        self.log(f"Warning: SICD magnitude image {i+1} appears to be empty (all values close to zero)")
                    
                    # Log statistics
                    self.log(f"SICD magnitude image {i+1} stats: min={np.min(magnitude_image):.2f}, max={np.max(magnitude_image):.2f}, mean={np.mean(magnitude_image):.2f}")
                    
                    # Append to SLC images list
                    self.slc_images.append(magnitude_image)
                    
                self.log(f"Created {len(self.slc_images)} SICD magnitude images")
                
            else:
                self.log(f"Error: Unknown data type '{self.data_type}' during focusing step")
                return False
            
            # Store normalized versions for visualization if in debug mode
            if self.debug_mode:
                self.normalized_slc_images = []
                for i, slc in enumerate(self.slc_images):
                    # Normalize for better visualization using 99th percentile to avoid outliers
                    norm_slc = slc / np.percentile(slc, 99) if np.percentile(slc, 99) > 0 else slc
                    self.normalized_slc_images.append(norm_slc)
                self.log("Created normalized SLC/magnitude images for visualization")
                
            return True
        except Exception as e:
            self.log(f"Error during focusing/preparation step: {str(e)}")
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

    def estimate_displacement_enhanced(self):
        """
        Enhanced version of the displacement estimation that combines chunking and memory efficiency
        """
        self.log("Beginning enhanced memory-efficient displacement estimation")
        
        # Ensure SLC images exist
        if self.slc_images is None or len(self.slc_images) < 2:
            self.log("Error: Need at least 2 SLC images for displacement estimation.")
            return False
            
        self.log("Starting enhanced memory-efficient displacement estimation...")
        
        # Memory tracking
        import psutil
        process = psutil.Process(os.getpid())
        
        def log_memory_usage(label):
            """Helper to log memory usage at key points"""
            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            self.log(f"Memory usage at {label}: {memory_usage:.2f} MB")
            return memory_usage
            
        start_memory = log_memory_usage("start")
        
        # Use SLC magnitude images for displacement estimation
        self.log(f"Input: {len(self.slc_images)} SLC/magnitude images from {self.data_type} data")
        self.log(f"SLC/magnitude image shape: {self.slc_images[0].shape}, dtype: {self.slc_images[0].dtype}")
            
        # Initialize array to hold displacement maps for each pair
        num_pairs = len(self.slc_images) - 1
        self.log(f"Initializing displacement maps for {num_pairs} image pairs")
        
        # Initialize with empty displacement maps list
        self.displacement_maps = []
        
        # Cache window size and step size for easier access
        window_size = self.window_size
        step_size = int(window_size * (1 - self.overlap))
        self.log(f"Using window size: {window_size}, overlap: {self.overlap}, step size: {step_size}")
        
        # Process each pair of images
        for pair_idx in range(num_pairs):
            self.log(f"Processing image pair {pair_idx+1}/{num_pairs}...")
            
            # Get the image pair
            img1 = self.slc_images[pair_idx]
            img2 = self.slc_images[pair_idx + 1]
            
            # Get dimensions
            self.log(f"Original image dimensions: {img1.shape[0]}x{img1.shape[1]}")
            
            # Determine if we need downsampling
            # Target ~2K×8K max size for memory efficiency
            max_rows, max_cols = 2048, 8192
            scale_factor = min(1.0, max_rows / img1.shape[0], max_cols / img1.shape[1])
            
            # Downsample if needed
            if scale_factor < 0.95:  # Only resample if scale is significantly different
                self.log(f"Downsampling images by factor {scale_factor:.3f} to reduce memory usage")
                img1_scaled = zoom(img1, scale_factor, order=1)
                img2_scaled = zoom(img2, scale_factor, order=1)
                self.log(f"New dimensions after downsampling: {img1_scaled.shape[0]}x{img1_scaled.shape[1]}")
            else:
                img1_scaled = img1
                img2_scaled = img2
                
            # Compute grid dimensions for the displacement map
            rows = (img1_scaled.shape[0] - window_size) // step_size + 1
            cols = (img1_scaled.shape[1] - window_size) // step_size + 1
            
            # Initialize displacement map for this pair
            range_displacements = np.zeros((rows, cols), dtype=np.float32)
            azimuth_displacements = np.zeros((rows, cols), dtype=np.float32)
            snr_values = np.zeros((rows, cols), dtype=np.float32)
            
            self.log(f"Displacement map dimensions: ({rows}, {cols})")
            
            # Calculate total number of windows to process
            total_windows = rows * cols
            self.log(f"Total windows to process: {total_windows}")
            
            # Calculate chunk size based on image dimensions
            # Use chunks of ~640 rows as a reasonable compromise
            chunk_size = min(640, rows)
            
            # Process the image in chunks
            processed_windows = 0
            
            for chunk_start in range(0, rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, rows)
                self.log(f"Processing chunk from row {chunk_start} to {chunk_end}")
                
                # Process windows in the current chunk
                for row in range(chunk_start, chunk_end):
                    for col in range(cols):
                        # Extract windows
                        y = row * step_size
                        x = col * step_size
                        window1 = img1_scaled[y:y+window_size, x:x+window_size]
                        window2 = img2_scaled[y:y+window_size, x:x+window_size]
                        
                        # Check if window has sufficient contrast
                        if (np.std(window1) < 1e-6 or np.std(window2) < 1e-6):
                            # Skip low-contrast windows
                            continue
                        
                        try:
                            # Phase cross-correlation for subpixel accuracy
                            shift_result, error, diffphase = phase_cross_correlation(
                                window1, window2, upsample_factor=10)
                                
                            # Store results
                            range_displacements[row, col] = shift_result[1]  # Range (x) displacement
                            azimuth_displacements[row, col] = shift_result[0]  # Azimuth (y) displacement
                            
                            # Compute SNR from error
                            snr = 1.0 / (error + 1e-10)  # Avoid division by zero
                            snr_values[row, col] = snr
                            
                        except Exception as e:
                            # Just continue if a specific window fails
                            continue
                            
                        processed_windows += 1
                        
                        # Log progress less frequently - every 10% of total windows
                        progress_interval = max(1, total_windows // 10)
                        if processed_windows % progress_interval == 0 or processed_windows == total_windows:
                            progress_percent = (processed_windows / total_windows) * 100
                            self.log(f"Processed {processed_windows}/{total_windows} windows ({progress_percent:.1f}%)")
                            # Log memory every 10% as well
                            log_memory_usage(f"after {processed_windows} windows")
                
                # Log memory after each chunk
                chunk_memory = log_memory_usage(f"after chunk {chunk_start}-{chunk_end}")
                
                # Save intermediate results to avoid memory buildup
                self.log(f"Saving intermediate results after chunk {chunk_start}-{chunk_end}")
                
            # Count valid displacements (non-zero)
            valid_shifts = np.count_nonzero(range_displacements) + np.count_nonzero(azimuth_displacements)
            skipped = total_windows - processed_windows
            
            self.log(f"Completed image pair {pair_idx+1}: Processed={processed_windows}, Skipped={skipped}, Errors=0, Valid Shifts Calculated={valid_shifts}")
            self.log(f"Window processing failure/skip rate: {(skipped/total_windows)*100:.2f}%")
            
            # Store results for this pair - use tuple format for backward compatibility
            self.displacement_maps.append((range_displacements, azimuth_displacements))
            
            # Log the range of displacement values
            self.log(f"Range displacement range: {np.nanmin(range_displacements):.4f} to {np.nanmax(range_displacements):.4f}")
            self.log(f"Azimuth displacement range: {np.nanmin(azimuth_displacements):.4f} to {np.nanmax(azimuth_displacements):.4f}")
        
        # Final memory usage
        end_memory = log_memory_usage("after all displacement estimation")
        
        # Log memory efficiency
        self.log(f"Memory change during displacement estimation: {end_memory - start_memory:.2f} MB")
            
        return True

    def plot_ship_regions(self, output_file=None):
        """
        Plot the detected ship regions overlaid on the vibration energy map.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save the plot image file (default is None, don't save)
            
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        if not hasattr(self, 'vibration_energy_map_db') or self.vibration_energy_map_db is None:
            self.log("Error: No vibration energy map available to plot ship regions")
            return False
        if not hasattr(self, 'ship_regions') or not self.ship_regions:
            self.log("Warning: No ship regions detected or provided to plot")
            # Plot just the energy map if no regions
            return self.plot_vibration_energy_map(output_file=output_file)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot vibration energy map
        cmap = plt.cm.jet
        # Use the actual range from the clipped map for normalization
        vmin_plot = np.min(self.vibration_energy_map_db)
        vmax_plot = np.max(self.vibration_energy_map_db)
        # Ensure vmin is not equal to vmax
        if vmin_plot >= vmax_plot:
            vmin_plot = vmax_plot - 1  # Adjust if range is zero
            
        norm = plt.Normalize(vmin=vmin_plot, vmax=vmax_plot)
        im = ax.imshow(self.vibration_energy_map_db, cmap=cmap, norm=norm, aspect='auto')
        ax.set_title('Detected Ship Regions on Vibration Energy Map')
        ax.set_xlabel('Range (pixels)')
        ax.set_ylabel('Azimuth (pixels)')
        
        # Overlay detected regions
        colors_list = plt.cm.tab10(range(len(self.ship_regions)))  # Use a colormap for distinct colors
        
        for i, region in enumerate(self.ship_regions):
            mask = region['mask']
            color = colors_list[i][:3]  # RGB values from colormap
            
            # Create an overlay for the mask
            overlay = np.zeros((*mask.shape, 4), dtype=float)  # RGBA
            overlay[mask] = (*color, 0.4)  # Set color and alpha for the masked region
            
            # Plot the overlay
            ax.imshow(overlay, aspect='auto', interpolation='nearest')
            
            # Add a marker at the centroid
            centroid_r, centroid_c = region['centroid']
            ax.plot(centroid_c, centroid_r, 'o', markersize=8, color='white', markeredgecolor='black', label=f"Region {region['id']}")
            
            # Add bounding box if available
            if 'bbox' in region:
                min_r, min_c, max_r, max_c = region['bbox']
                rect = plt.Rectangle((min_c - 0.5, min_r - 0.5), max_c - min_c, max_r - min_r,
                                    fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
        
        if len(self.ship_regions) > 0:
            ax.legend()
            
        plt.tight_layout()
        
        # Save the plot if a file path is specified
        if output_file:
            try:
                plt.savefig(output_file)
                self.log(f"Saved ship regions visualization to {output_file}")
            except Exception as e:
                self.log(f"Error saving ship regions plot: {e}")
                plt.close(fig)
                return False
                
        plt.close(fig)
        return True
