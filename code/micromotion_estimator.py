import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import shift
from scipy.fft import fft, fftfreq
from skimage.registration import phase_cross_correlation
import matplotlib.colors as colors
from matplotlib.patches import Polygon

class ShipMicroMotionEstimator:
    """
    Class for estimating micro-motion of ships from SAR images using pixel tracking
    Based on the paper: "Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in 
    Cosmo-Skymed Synthetic Aperture Radar Data—An Operative Assessment"
    """
    
    def __init__(self, num_subapertures=7, window_size=64, overlap=0.5):
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
        """
        self.num_subapertures = num_subapertures
        self.window_size = window_size
        self.overlap = overlap
        self.subapertures = None
        self.displacement_maps = None
        self.time_series = None
        self.frequency_spectra = None
        self.vibration_energy_map = None
        self.ship_regions = None
        
    def load_data(self, cphd_file_path, channel_index=0):
        """
        Load data from a CPHD file
        
        Parameters
        ----------
        cphd_file_path : str
            Path to the CPHD file
        channel_index : int, optional
            Index of the channel to read (default is 0)
            
        Returns
        -------
        bool
            True if data was loaded successfully, False otherwise
        """
        try:
            # For actual implementation, use sarpy library to read CPHD data
            # Here we'll use a placeholder for demonstration
            print(f"Loading data from {cphd_file_path}...")
            
            # Create synthetic data for demonstration if CPHD file doesn't exist
            if not os.path.exists(cphd_file_path):
                print("CPHD file not found, creating synthetic data for demonstration...")
                from test_estimator import create_synthetic_data
                self.subapertures, pvp, _ = create_synthetic_data(
                    rows=512, cols=512, num_subapertures=self.num_subapertures
                )
                
                # Generate SLC images with more pronounced features
                self.slc_images = []
                for subap in self.subapertures:
                    # Use magnitude of the complex data as the SLC image
                    slc = np.abs(subap)
                    # Enhance contrast to make features more distinct
                    slc = np.power(slc, 0.5)  # Square root to enhance contrast
                    self.slc_images.append(slc)
                
                return True
            
            # If file exists, use sarpy to read it
            try:
                from sarpy.io.phase_history.cphd import CPHDReader
                reader = CPHDReader(cphd_file_path)
                
                # Get metadata
                metadata = reader.cphd_meta
                
                # Read phase data for the specified channel
                data = reader.read_chip()
                
                # Get PVP data
                pvp = reader.read_pvp_block()
                
                # Split into sub-apertures
                self.subapertures = self._split_aperture(data, self.num_subapertures)
                
                # Focus each sub-aperture to generate SLC images
                self.slc_images = self._focus_subapertures(self.subapertures, pvp)
                
                return True
            except Exception as e:
                print(f"Error reading CPHD file with sarpy: {e}")
                return False
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _split_aperture(self, data, num_subapertures):
        """
        Split the phase history data into multiple Doppler sub-apertures
        
        Parameters
        ----------
        data : numpy.ndarray
            Phase history data
        num_subapertures : int
            Number of sub-apertures to create
            
        Returns
        -------
        list
            List of numpy arrays containing the sub-aperture data
        """
        # In a real implementation, this would use the Doppler spectrum
        # to split the data into sub-apertures. For now, we'll just
        # create copies with slight modifications to simulate time variation.
        
        subapertures = []
        for i in range(num_subapertures):
            # Create a copy of the data with slight modifications
            subap = data.copy()
            subapertures.append(subap)
        
        return subapertures
    
    def _focus_subapertures(self, subapertures, pvp):
        """
        Focus each sub-aperture to generate SLC images
        This is a simplified implementation and would need to be replaced with
        actual SAR focusing algorithm in a production environment
        
        Parameters
        ----------
        subapertures : list
            List of numpy arrays containing the sub-aperture data
        pvp : dict
            Dictionary containing PVP (Per Vector Parameters) data
            
        Returns
        -------
        list
            List of numpy arrays containing the focused SLC images
        """
        # In a real implementation, this would use the PVP data to properly focus
        # each sub-aperture. For now, we'll just use the magnitude of the data
        # as a placeholder for the focused images.
        slc_images = []
        for subap in subapertures:
            # Apply a simple FFT as a placeholder for proper focusing
            focused = np.fft.fft2(subap)
            slc_images.append(np.abs(focused))
        
        return slc_images
    
    def estimate_displacement(self):
        """
        Estimate displacement between adjacent sub-apertures using sub-pixel offset tracking
        
        Returns
        -------
        bool
            True if displacement was estimated successfully, False otherwise
        """
        if self.slc_images is None or len(self.slc_images) < 2:
            print("Error: No SLC images available")
            return False
        
        # Initialize displacement maps
        self.displacement_maps = []
        
        # Calculate step size based on window size and overlap
        step = int(self.window_size * (1 - self.overlap))
        
        # For each pair of adjacent sub-apertures
        for i in range(len(self.slc_images) - 1):
            ref_image = self.slc_images[i]
            sec_image = self.slc_images[i + 1]
            
            # Get image dimensions
            rows, cols = ref_image.shape
            
            # Initialize displacement map for this pair
            range_shifts = np.zeros((rows // step, cols // step))
            azimuth_shifts = np.zeros((rows // step, cols // step))
            
            # For each window position
            for r in range(0, rows - self.window_size, step):
                for c in range(0, cols - self.window_size, step):
                    # Extract windows from both images
                    ref_window = ref_image[r:r+self.window_size, c:c+self.window_size]
                    sec_window = sec_image[r:r+self.window_size, c:c+self.window_size]
                    
                    # Calculate sub-pixel shift using phase cross-correlation
                    try:
                        shift, error, diffphase = phase_cross_correlation(
                            ref_window, sec_window, upsample_factor=100
                        )
                        
                        # Store shifts in the displacement map
                        row_idx = r // step
                        col_idx = c // step
                        range_shifts[row_idx, col_idx] = shift[0]
                        azimuth_shifts[row_idx, col_idx] = shift[1]
                    except Exception as e:
                        print(f"Error calculating shift at position ({r}, {c}): {e}")
            
            # Store displacement maps for this pair
            self.displacement_maps.append((range_shifts, azimuth_shifts))
        
        return True
    
    def analyze_time_series(self, measurement_points):
        """
        Analyze time series of displacements at specified measurement points
        
        Parameters
        ----------
        measurement_points : list
            List of (row, col) tuples specifying measurement points
            
        Returns
        -------
        bool
            True if time series was analyzed successfully, False otherwise
        """
        if self.displacement_maps is None or len(self.displacement_maps) == 0:
            print("Error: No displacement maps available")
            return False
        
        # Initialize time series and frequency spectra
        self.time_series = {
            'range': {}, 
            'azimuth': {}
        }
        self.frequency_spectra = {
            'range': {}, 
            'azimuth': {}
        }
        
        # Calculate step size based on window size and overlap
        step = int(self.window_size * (1 - self.overlap))
        
        # For each measurement point
        for idx, (row, col) in enumerate(measurement_points):
            # Convert to displacement map indices
            map_row = row // step
            map_col = col // step
            
            # Extract time series for range and azimuth
            range_series = []
            azimuth_series = []
            
            for range_map, azimuth_map in self.displacement_maps:
                if map_row < range_map.shape[0] and map_col < range_map.shape[1]:
                    range_series.append(range_map[map_row, map_col])
                    azimuth_series.append(azimuth_map[map_row, map_col])
            
            # Store time series
            self.time_series['range'][idx] = np.array(range_series)
            self.time_series['azimuth'][idx] = np.array(azimuth_series)
            
            # Calculate frequency spectra using FFT
            if len(range_series) > 0:
                # Sampling frequency (assuming uniform time steps)
                # In a real implementation, this would be derived from the PVP data
                fs = 10.0  # Hz, placeholder value
                
                # Calculate FFT
                range_fft = np.abs(fft(range_series))
                azimuth_fft = np.abs(fft(azimuth_series))
                
                # Calculate frequency bins
                n = len(range_series)
                freq = fftfreq(n, 1/fs)[:n//2]
                
                # Store frequency spectra (positive frequencies only)
                self.frequency_spectra['range'][idx] = (freq, range_fft[:n//2])
                self.frequency_spectra['azimuth'][idx] = (freq, azimuth_fft[:n//2])
        
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
    
    def plot_results(self, measurement_point_idx, output_dir=None):
        """
        Plot time series and frequency spectra for a specific measurement point
        
        Parameters
        ----------
        measurement_point_idx : int
            Index of the measurement point to plot
        output_dir : str, optional
            Directory to save plots (if None, plots are displayed but not saved)
            
        Returns
        -------
        bool
            True if plots were created successfully, False otherwise
        """
        if (self.time_series is None or 
            measurement_point_idx not in self.time_series['range'] or 
            measurement_point_idx not in self.time_series['azimuth']):
            print(f"Error: No data available for measurement point {measurement_point_idx}")
            return False
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot range time series
        axs[0, 0].plot(self.time_series['range'][measurement_point_idx])
        axs[0, 0].set_title(f'Range Displacement Time Series (Point {measurement_point_idx})')
        axs[0, 0].set_xlabel('Time (samples)')
        axs[0, 0].set_ylabel('Displacement (pixels)')
        axs[0, 0].grid(True)
        
        # Plot azimuth time series
        axs[0, 1].plot(self.time_series['azimuth'][measurement_point_idx])
        axs[0, 1].set_title(f'Azimuth Displacement Time Series (Point {measurement_point_idx})')
        axs[0, 1].set_xlabel('Time (samples)')
        axs[0, 1].set_ylabel('Displacement (pixels)')
        axs[0, 1].grid(True)
        
        # Plot range frequency spectrum
        freq, spectrum = self.frequency_spectra['range'][measurement_point_idx]
        axs[1, 0].plot(freq, spectrum)
        axs[1, 0].set_title(f'Range Frequency Spectrum (Point {measurement_point_idx})')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Amplitude')
        axs[1, 0].grid(True)
        
        # Plot azimuth frequency spectrum
        freq, spectrum = self.frequency_spectra['azimuth'][measurement_point_idx]
        axs[1, 1].plot(freq, spectrum)
        axs[1, 1].set_title(f'Azimuth Frequency Spectrum (Point {measurement_point_idx})')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Amplitude')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save or display the plot
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'point_{measurement_point_idx}_results.png'))
            plt.close()
        else:
            plt.show()
        
        return True
    
    def plot_vibration_energy_map(self, output_dir=None):
        """
        Plot vibration energy map with ship regions
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save plots (if None, plots are displayed but not saved)
            
        Returns
        -------
        bool
            True if plots were created successfully, False otherwise
        """
        if self.vibration_energy_map_db is None:
            print("Error: No vibration energy map available")
            return False
        
        # Create figure with 1x2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot SLC image on the left
        if self.slc_images is not None and len(self.slc_images) > 0:
            # Use the first SLC image
            slc_image = self.slc_images[0]
            axs[0].imshow(slc_image, cmap='gray')
            axs[0].set_title('SLC Image')
            axs[0].set_xlabel('Range (pixels)')
            axs[0].set_ylabel('Azimuth (pixels)')
        else:
            # Create a blank image
            axs[0].imshow(np.zeros_like(self.vibration_energy_map_db), cmap='gray')
            axs[0].set_title('SLC Image (Not Available)')
            axs[0].set_xlabel('Range (pixels)')
            axs[0].set_ylabel('Azimuth (pixels)')
        
        # Plot vibration energy map on the right
        cmap = plt.cm.jet
        norm = colors.Normalize(vmin=-25, vmax=0)
        im = axs[1].imshow(self.vibration_energy_map_db, cmap=cmap, norm=norm)
        axs[1].set_title('SLC ROI Vibration Energy (dB)')
        axs[1].set_xlabel('Range (pixels)')
        axs[1].set_ylabel('Azimuth (pixels)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axs[1])
        cbar.set_label('Vibration Energy (dB)')
        
        # Add ship region labels if available
        if self.ship_regions is not None:
            for region in self.ship_regions:
                # Add label to both plots
                region_id = region['id']
                centroid = region['centroid']
                
                # Add label to SLC image
                axs[0].text(centroid[1], centroid[0], str(region_id), 
                           color='white', fontsize=12, ha='center', va='center',
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                
                # Add label to vibration energy map
                axs[1].text(centroid[1], centroid[0], str(region_id), 
                           color='white', fontsize=12, ha='center', va='center',
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                
                # Add arrows pointing to the regions
                # For SLC image
                arrow_length = 30
                arrow_angle = np.random.uniform(0, 2*np.pi)  # Random angle for variety
                arrow_dx = arrow_length * np.cos(arrow_angle)
                arrow_dy = arrow_length * np.sin(arrow_angle)
                arrow_start_x = centroid[1] + arrow_dx
                arrow_start_y = centroid[0] + arrow_dy
                
                axs[0].annotate('', xy=(centroid[1], centroid[0]), 
                               xytext=(arrow_start_x, arrow_start_y),
                               arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
                
                # For vibration energy map
                axs[1].annotate('', xy=(centroid[1], centroid[0]), 
                               xytext=(arrow_start_x, arrow_start_y),
                               arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
        
        plt.tight_layout()
        
        # Save or display the plot
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'vibration_energy_map.png'))
            plt.close()
        else:
            plt.show()
        
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
