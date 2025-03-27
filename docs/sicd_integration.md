# SICD Data Integration Plan

## Overview

This document outlines the plan and code modifications required to integrate support for SICD (Sensor Independent Complex Data) NITF files into the Ship Micro-Motion Estimator application, alongside the existing CPHD support. The goal is to allow users to upload either CPHD or SICD files and have the application process them appropriately.

## Approach

1.  **File Type Detection:** The application will determine the input data type (CPHD or SICD) based on the file extension or potentially by inspecting the file using a library like `sarpy`.
2.  **Conditional Logic in Estimator:** The `ShipMicroMotionEstimator` class will be modified to:
    *   Store the detected data type (`'cphd'` or `'sicd'`).
    *   Use different logic in `load_data` to read the appropriate data (phase history for CPHD, complex image for SICD).
    *   Use different logic for sub-aperture creation (`_split_aperture_cphd` vs. `_split_aperture_sicd`).
    *   **Skip the focusing step (`_focus_subapertures`) entirely for SICD data**, as it's already focused. The step will still be called but will perform a simple operation (like taking magnitude) for consistency in the pipeline if needed.
3.  **SICD Sub-Aperture Handling:** For SICD, "sub-apertures" will be created by splitting the complex image along the azimuth (time) dimension. This provides multiple temporal snapshots from the single focused image, mimicking the CPHD sub-aperture concept for the subsequent displacement analysis. If only a single SICD image is provided and splitting is not desired or possible, the sub-aperture step might be bypassed, limiting the analysis. (This implementation assumes splitting is desired).
4.  **Pipeline Consistency:** The `estimate_displacement_enhanced` function expects magnitude images. For the SICD path, the magnitude (`np.abs()`) will be taken *after* splitting the complex image into sub-images (within the modified `focus_subapertures` step).
5.  **Application Workflow (`app.py`):** The main processing route will be updated to correctly call the estimator's methods based on the detected data type, ensuring the focusing step is handled conditionally.

## Detailed Code Changes

### 1. `micromotion_estimator.py` Modifications

*   Add `data_type` attribute.
*   Update `load_data` to handle both CPHD and SICD using `sarpy`.
*   Rename `_split_aperture` to `_split_aperture_cphd`.
*   Add `_split_aperture_sicd` for image splitting.
*   Update `create_subapertures` to use conditional logic.
*   Update `focus_subapertures` to skip focusing for SICD and prepare magnitude images.

```python:code/micromotion_estimator.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import shift, zoom # Added zoom
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from skimage.registration import phase_cross_correlation
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from skimage.transform import warp, AffineTransform
# Removed: from cphd_reader import split_aperture - splitting logic now internal
from sarpy.io.complex.sicd_elements.SICD import SICDType # For SICD metadata typing
from sarpy.io.general.base import BaseReader # For type checking

class ShipMicroMotionEstimator:
    # ... (existing docstring) ...
    def __init__(self, num_subapertures=7, window_size=64, overlap=0.5, debug_mode=False, log_callback=None):
        # ... (existing parameters) ...
        self.num_subapertures = num_subapertures
        self.window_size = window_size
        self.overlap = overlap
        self.debug_mode = debug_mode
        self.log_callback = log_callback
        self.data_type = None # Added: 'cphd' or 'sicd'
        self.subapertures = None # Will store complex data for both types before focusing/abs
        self.subaperture_times = None # Added: To store timing info
        self.displacement_maps = None
        self.time_series = None
        self.frequency_spectra = None
        self.vibration_energy_map = None
        self.vibration_energy_map_db = None # Added: To store dB map
        self.ship_regions = None
        self.data_loaded = False

        # Debug mode attributes
        self.raw_data = None # For CPHD phase history
        self.raw_complex_data = None # Added: For SICD complex image data
        self.pvp = None # Added: Store PVP for CPHD
        self.sicd_meta = None # Added: Store SICD metadata structure
        self.raw_data_image = None # Magnitude image for visualization
        self.slc_images = None # Will store magnitude images for displacement estimation
        self.snr_maps = {}
        self.coherence_maps = {}

        self.last_error = None

    # ... (log method remains the same) ...

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
            self.data_type = 'cphd' # Assume demo is CPHD-like
            from test_estimator import create_synthetic_data
            # Demo data creates 'subapertures' directly (magnitude)
            # We need complex data ideally, or adapt the flow
            # For now, let's assume demo creates complex subapertures
            complex_subaps, pvp, _ = create_synthetic_data(
                rows=512, cols=512, num_subapertures=self.num_subapertures, output_complex=True
            )
            self.subapertures = complex_subaps # Store complex subapertures
            self.pvp = pvp
            self.subaperture_times = np.linspace(0, 1, self.num_subapertures) # Synthetic times
            # Create a representative raw_data_image (e.g., from first subap)
            if self.subapertures:
                 self.raw_data_image = np.abs(self.subapertures[0])
            self.data_loaded = True
            self.log("Synthetic demo data loaded.")
            return True
        elif not os.path.exists(file_path):
             self.log(f"Error: File not found at {file_path}")
             return False
        else:
            try:
                from sarpy.io import open as sarpy_open

                self.log(f"Opening file with sarpy: {file_path}")
                reader = sarpy_open(file_path)

                if not isinstance(reader, BaseReader):
                     self.log(f"Error: Sarpy could not open the file or returned an unexpected type: {type(reader)}")
                     return False

                # Determine data type based on reader type or metadata
                # CPHD readers often have 'cphd' in their name or specific metadata
                # SICD readers usually provide a SICD metadata structure
                reader_type_name = type(reader).__name__
                self.log(f"Sarpy reader type: {reader_type_name}")

                has_cphd_meta = hasattr(reader, 'cphd_meta') and reader.cphd_meta is not None
                has_sicd_meta = hasattr(reader, 'sicd_meta') and reader.sicd_meta is not None

                if has_cphd_meta or 'cphd' in reader_type_name.lower():
                    self.data_type = 'cphd'
                    self.log("Detected CPHD data type.")
                    self.sicd_meta = None

                    # --- CPHD Loading Logic ---
                    self.log("Reading CPHD phase history data...")
                    # Assuming read_chip reads the phase history data for CPHD
                    # Use index=channel_index if reader supports it
                    try:
                        if hasattr(reader, 'read_chip'):
                             data_indices = list(reader.get_data_indices())
                             if not data_indices:
                                 self.log("Error: Reader has no data indices.")
                                 return False
                             read_index = data_indices[min(channel_index, len(data_indices)-1)]
                             self.log(f"Reading CPHD chip for index: {read_index}")
                             self.raw_data = reader.read_chip(index=read_index) # Store raw phase history
                        else:
                             self.log("Error: CPHD reader does not have read_chip method.")
                             return False
                    except Exception as e:
                         self.log(f"Error reading CPHD data chip: {e}")
                         return False

                    self.log(f"Raw CPHD data shape: {self.raw_data.shape}")
                    self.raw_data_image = np.abs(self.raw_data) # For visualization

                    # Read PVP data for timing
                    self.log("Reading PVP data...")
                    try:
                        # Simplified PVP reading - assumes read_pvp() returns a dict-like structure
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

                elif has_sicd_meta or 'sicd' in reader_type_name.lower():
                    self.data_type = 'sicd'
                    self.log("Detected SICD data type.")
                    self.raw_data = None # No raw phase history for SICD
                    self.pvp = None

                    # --- SICD Loading Logic ---
                    self.log("Reading SICD complex image data...")
                    # Assuming read_chip reads the complex image data for SICD
                    try:
                        if hasattr(reader, 'read_chip'):
                             data_indices = list(reader.get_data_indices())
                             if not data_indices:
                                 self.log("Error: Reader has no data indices.")
                                 return False
                             read_index = data_indices[min(channel_index, len(data_indices)-1)]
                             self.log(f"Reading SICD chip for index: {read_index}")
                             self.raw_complex_data = reader.read_chip(index=read_index) # Store complex image
                        else:
                             self.log("Error: SICD reader does not have read_chip method.")
                             return False
                    except Exception as e:
                         self.log(f"Error reading SICD data chip: {e}")
                         return False

                    self.log(f"Raw SICD complex data shape: {self.raw_complex_data.shape}")
                    self.raw_data_image = np.abs(self.raw_complex_data) # For visualization

                    # Store SICD metadata
                    self.sicd_meta = reader.sicd_meta
                    if isinstance(self.sicd_meta, list): # Handle multiple SICDs in a file
                        self.sicd_meta = self.sicd_meta[min(channel_index, len(self.sicd_meta)-1)]

                    if not isinstance(self.sicd_meta, SICDType):
                         self.log(f"Warning: Expected SICDType metadata, but got {type(self.sicd_meta)}. Timing information might be unavailable.")
                         self.sicd_meta = None # Clear if not the expected type

                    self.log("SICD data loaded successfully.")

                else:
                    self.log(f"Error: Could not determine data type (CPHD or SICD) for reader {reader_type_name}.")
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
        """
        # ... (Keep existing logic from the original _split_aperture) ...
        # Ensure it uses self.pvp['TxTime'] for timing if available
        num_pulses, num_samples = data.shape
        if num_subapertures <= 0 or num_subapertures > num_pulses:
            raise ValueError("Invalid number of sub-apertures for CPHD")
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")

        # Calculate subaperture size and step
        # Ensure subaperture_size is at least 1
        subaperture_size = max(1, int(np.floor(num_pulses / (num_subapertures - (num_subapertures - 1) * overlap))))
        step = max(1, int(subaperture_size * (1 - overlap)))
        self.log(f"CPHD Splitting: num_pulses={num_pulses}, num_subaps={num_subapertures}, overlap={overlap} -> subap_size={subaperture_size}, step={step}")


        subapertures = []
        subaperture_times = []

        # Get timing information
        if hasattr(self, 'pvp') and self.pvp and 'TxTime' in self.pvp and self.pvp['TxTime'] is not None and len(self.pvp['TxTime']) == num_pulses:
            times = self.pvp['TxTime']
            self.log("Using TxTime from PVP for subaperture timing.")
        else:
            self.log("Warning: Valid TxTime not found in PVP or length mismatch. Using synthetic linear timing.")
            times = np.linspace(0, 1, num_pulses) # Synthetic timing

        start_idx = 0
        for i in range(num_subapertures):
            end_idx = start_idx + subaperture_size
            # Ensure the last subaperture reaches the end
            if i == num_subapertures - 1:
                end_idx = num_pulses
            # Prevent exceeding bounds
            end_idx = min(end_idx, num_pulses)
            # Ensure start index is valid
            start_idx = min(start_idx, num_pulses -1)
            # Ensure we have at least one sample
            if end_idx <= start_idx:
                 if i > 0: # If not the first, try to take the last sample
                     start_idx = end_idx - 1
                 else: # If first, cannot proceed
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
        """
        if data is None:
            raise ValueError("SICD complex data is not loaded.")

        num_azimuth, num_range = data.shape # Assuming azimuth is axis 0
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
            num_subapertures = num_azimuth # Cannot have more subaps than lines

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
        duration = end_time - start_time if start_time is not None else 1.0 # Default duration 1.0 if no timing
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
            start_idx = min(start_idx, num_azimuth -1)
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
        """Helper to extract start and end time from SICD metadata."""
        if self.sicd_meta and isinstance(self.sicd_meta, SICDType):
            try:
                # Accessing timeline information - structure might vary slightly
                if self.sicd_meta.Timeline is not None:
                    start_time = self.sicd_meta.Timeline.CollectStart
                    end_time = self.sicd_meta.Timeline.CollectDuration + start_time
                    if start_time is not None and end_time is not None:
                         self.log(f"Extracted SICD timing: Start={start_time}, End={end_time}")
                         return start_time, end_time
                # Fallback using IPP if Timeline is incomplete
                if self.sicd_meta.Timeline is not None and self.sicd_meta.Timeline.IPP is not None and self.sicd_meta.Timeline.IPP.Set:
                     ipp_set = self.sicd_meta.Timeline.IPP.Set[0]
                     start_idx = ipp_set.IPPStart
                     end_idx = ipp_set.IPPEnd
                     num_ipps = (end_idx - start_idx) + 1
                     t_rate = ipp_set.IPPPoly[0] # Assuming zero-order polynomial (constant rate)
                     if t_rate > 0 and num_ipps > 0:
                          duration = num_ipps * t_rate
                          # Estimate start time (might need reference time from metadata)
                          start_time = self.sicd_meta.Timeline.CollectStart if self.sicd_meta.Timeline.CollectStart is not None else 0.0
                          end_time = start_time + duration
                          self.log(f"Estimated SICD timing from IPP: Start={start_time}, End={end_time}")
                          return start_time, end_time

            except AttributeError as e:
                self.log(f"Could not access expected SICD metadata fields for timing: {e}")
            except Exception as e:
                 self.log(f"Error extracting SICD timing: {e}")
        self.log("Warning: Could not determine SICD timing information. Using default [0, 1].")
        return 0.0, 1.0 # Default timing

    def create_subapertures(self):
        """Create subapertures based on the loaded data type (CPHD or SICD)"""
        self.log("Starting subaperture creation...")

        if not self.data_loaded:
            self.log("Error: Data not loaded.")
            return False

        # If subapertures already exist (e.g., from demo data), skip
        if hasattr(self, 'subapertures') and self.subapertures and len(self.subapertures) > 0:
            self.log(f"Subapertures already exist ({len(self.subapertures)} found), reusing them.")
            # Ensure subaperture_times also exists if reusing
            if not hasattr(self, 'subaperture_times') or self.subaperture_times is None or len(self.subaperture_times) != len(self.subapertures):
                 self.log("Warning: Reusing subapertures but timing info is missing or mismatched. Generating synthetic timing.")
                 self.subaperture_times = np.linspace(0, 1, len(self.subapertures))
            return True

        try:
            if self.data_type == 'cphd':
                if self.raw_data is None:
                    self.log("Error: Raw CPHD data not available.")
                    return False
                self.log(f"Creating {self.num_subapertures} CPHD subapertures with overlap {self.overlap}...")
                self.subapertures, self.subaperture_times = self._split_aperture_cphd(
                    self.raw_data, self.num_subapertures, self.overlap
                )
                self.log(f"Created {len(self.subapertures)} CPHD sub-apertures.")

            elif self.data_type == 'sicd':
                if self.raw_complex_data is None:
                    self.log("Error: Raw SICD complex data not available.")
                    return False
                self.log(f"Creating {self.num_subapertures} SICD sub-images with overlap {self.overlap}...")
                self.subapertures, self.subaperture_times = self._split_aperture_sicd(
                    self.raw_complex_data, self.num_subapertures, self.overlap
                )
                self.log(f"Created {len(self.subapertures)} SICD sub-images.")

            else:
                self.log(f"Error: Unknown or unsupported data type '{self.data_type}'")
                return False

            # Validate created subapertures
            if not self.subapertures:
                 self.log("Error: Subaperture creation resulted in an empty list.")
                 return False

            for i, subap in enumerate(self.subapertures):
                self.log(f"Subaperture {i+1} shape: {subap.shape}")
                if np.all(np.abs(subap) < 1e-10):
                    self.log(f"Warning: Subaperture {i+1} appears to be empty (all zeros)")

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
            self.log(f"SLC magnitude images already exist ({len(self.slc_images)} found), reusing them.")
            return True

        if not hasattr(self, 'subapertures') or not self.subapertures or len(self.subapertures) == 0:
            self.log("Error: No subapertures available to focus or prepare.")
            return False

        try:
            self.slc_images = [] # Initialize list for magnitude images

            if self.data_type == 'cphd':
                self.log(f"Focusing {len(self.subapertures)} CPHD subapertures...")
                # **********************************************************************
                # ** WARNING: Placeholder Focusing Implementation **
                # Replace np.abs() with a proper SAR focusing algorithm for CPHD.
                # **********************************************************************
                self.log("WARNING: Using placeholder focusing (np.abs) for CPHD. Replace with proper SAR focusing algorithm.")

                for i, subap in enumerate(self.subapertures):
                    self.log(f"Focusing CPHD subaperture {i+1}/{len(self.subapertures)}...")
                    # --- Replace this line with actual focusing ---
                    focused_slc = np.abs(subap)
                    # ---------------------------------------------
                    self.slc_images.append(focused_slc)
                    self.log(f"CPHD SLC image {i+1} stats: min={np.min(focused_slc):.2f}, max={np.max(focused_slc):.2f}, mean={np.mean(focused_slc):.2f}")

            elif self.data_type == 'sicd':
                self.log(f"Preparing {len(self.subapertures)} SICD sub-images (taking magnitude)...")
                # SICD data is already focused. We just need the magnitude images
                # for the current displacement estimation method.
                for i, complex_sub_image in enumerate(self.subapertures):
                    self.log(f"Taking magnitude of SICD sub-image {i+1}/{len(self.subapertures)}...")
                    magnitude_image = np.abs(complex_sub_image)
                    self.slc_images.append(magnitude_image)
                    self.log(f"SICD magnitude image {i+1} stats: min={np.min(magnitude_image):.2f}, max={np.max(magnitude_image):.2f}, mean={np.mean(magnitude_image):.2f}")

            else:
                self.log(f"Error: Unknown data type '{self.data_type}' during focusing step.")
                return False

            self.log(f"Generated {len(self.slc_images)} SLC/magnitude images.")

            # Validate generated images
            if not self.slc_images:
                 self.log("Error: SLC/magnitude image list is empty after processing.")
                 return False
            for i, slc in enumerate(self.slc_images):
                 if np.all(slc < 1e-10):
                      self.log(f"Warning: SLC/magnitude image {i+1} appears empty.")

            return True

        except Exception as e:
            self.log(f"Error during focusing/preparation step: {str(e)}")
            import traceback
            trace = traceback.format_exc()
            self.log(trace)
            self.last_error = f"{str(e)}\n\n{trace}"
            return False

    # --- estimate_displacement_enhanced ---
    # This method should work as is, provided self.slc_images contains
    # the magnitude images correctly prepared by focus_subapertures.
    # Consider adding a check at the beginning:
    def estimate_displacement_enhanced(self):
        # ... (existing docstring) ...
        import gc
        # ... (rest of imports) ...

        self.log("Starting enhanced memory-efficient displacement estimation...")
        log_memory_usage("start displacement estimation") # Renamed label

        # Add check for slc_images
        if self.slc_images is None or not isinstance(self.slc_images, list) or len(self.slc_images) < 2:
            self.log("Error: SLC magnitude images not available or insufficient for displacement estimation.")
            return False
        # Add check for image content type (should be real/magnitude)
        if np.iscomplexobj(self.slc_images[0]):
             self.log("Error: Displacement estimation expects magnitude images, but received complex data.")
             return False

        # ... (rest of the existing estimate_displacement_enhanced method) ...


    # --- analyze_time_series ---
    # Needs self.subaperture_times to be correctly populated for both CPHD and SICD paths.
    def analyze_time_series(self, measurement_points):
        # ... (existing docstring) ...
        if not hasattr(self, 'displacement_maps') or not self.displacement_maps:
            self.log("Error: No displacement maps available for time series analysis.") # Changed print to log
            return False
        # Add check for subaperture_times
        if not hasattr(self, 'subaperture_times') or self.subaperture_times is None or len(self.subaperture_times) < 2:
             self.log("Error: Subaperture timing information not available or insufficient for frequency analysis.")
             # Decide whether to proceed with synthetic timing or fail
             # For now, let's try to proceed but log a clear warning
             self.log("Warning: Proceeding with synthetic time delta for frequency analysis.")
             fs = 1.0 # Default sampling frequency
        else:
             # Calculate sampling frequency from subaperture_times
             delta_t = np.mean(np.diff(self.subaperture_times))
             if delta_t <= 0:
                  self.log(f"Warning: Calculated non-positive time delta ({delta_t}). Using default fs=1.0.")
                  fs = 1.0
             else:
                  fs = 1.0 / delta_t
                  self.log(f"Calculated sampling frequency: {fs:.2f} Hz from mean delta_t={delta_t:.4f} s")


        self.time_series = {'range': {}, 'azimuth': {}}
        self.frequency_spectra = {'range': {}, 'azimuth': {}}
        step = int(self.window_size * (1 - self.overlap))

        # Check if displacement maps have expected structure
        if not isinstance(self.displacement_maps, list) or len(self.displacement_maps) == 0 or not isinstance(self.displacement_maps[0], tuple) or len(self.displacement_maps[0]) != 2:
             self.log("Error: Displacement maps have unexpected structure.")
             return False


        for idx, (row, col) in enumerate(measurement_points):
            # ... (rest of the loop logic) ...

            # Calculate frequency spectra
            if len(range_series) > 1: # Ensure there's enough data for FFT
                # Use the calculated sampling frequency 'fs'
                n = len(range_series)
                freq = fftfreq(n, 1 / fs)[:n // 2] # Use calculated fs
                range_fft = np.abs(fft(range_series))[:n // 2]
                azimuth_fft = np.abs(fft(azimuth_series))[:n // 2]

                self.frequency_spectra['range'][idx] = (freq, range_fft)
                self.frequency_spectra['azimuth'][idx] = (freq, azimuth_fft)
            else:
                 self.log(f"Warning: Insufficient time series data ({len(range_series)} points) at point {idx} to calculate spectrum.")
                 # Store empty spectrum?
                 self.frequency_spectra['range'][idx] = (np.array([]), np.array([]))
                 self.frequency_spectra['azimuth'][idx] = (np.array([]), np.array([]))


        self.log("Time series analysis completed.")
        return True

    # ... (rest of the methods: calculate_vibration_energy_map, plot_ship_regions, etc.) ...
    # These should work correctly if displacement_maps are generated properly.

```

### 2. `app.py` Modifications

*   Update the main `process` function to handle the conditional workflow based on `estimator.data_type`.
*   Ensure file extension checking includes `.nitf` or `.ntf` for SICD.

```python:code/app.py
# ... (existing imports) ...
import time
import psutil
import fnmatch # For more flexible file extension matching

# ... (logging setup) ...

# Allowed file extensions
# Use fnmatch for flexibility (e.g., allow .NTF, .nitf)
ALLOWED_CPHD_PATTERNS = ['*.cphd', '*.CPHD']
ALLOWED_SICD_PATTERNS = ['*.ntf', '*.NTF', '*.nitf', '*.NITF']
ALLOWED_PATTERNS = ALLOWED_CPHD_PATTERNS + ALLOWED_SICD_PATTERNS

def allowed_file(filename):
    for pattern in ALLOWED_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

# ... (list_uploaded_files remains the same, uses allowed_file) ...
# ... (fig_to_base64, create_results_folder, update_processing_status, log_to_processing remain the same) ...

# ... (ALGORITHM_STEPS, STEP_DURATIONS might need adjustment if SICD path is faster/slower) ...

# ... (index, check_status, debug_playground, run_debug_step remain largely the same) ...
# Note: run_debug_step might need updates if you want fine-grained SICD debugging

@app.route('/process', methods=['POST'])
def process():
    """
    Process uploaded CPHD/SICD file or demo data and analyze micro-motion.
    """
    # ... (setup result_dir, logging, detailed_log_path) ...

    logger.info(f"Starting new processing run, results will be saved to {result_dir}")
    update_processing_status('starting', 0, 'Initializing processing',
                           console_log=f"Starting new processing run, results directory: {result_dir}")

    try:
        # ... (get parameters: num_subapertures, window_size, overlap, detector params) ...

        # --- Instantiate Estimator and Detector ---
        estimator = ShipMicroMotionEstimator(num_subapertures=num_subapertures,
                                             window_size=window_size,
                                             overlap=overlap,
                                             debug_mode=True,
                                             log_callback=append_to_detailed_log)

        detector = ShipRegionDetector(min_region_size=min_region_size,
                                      energy_threshold=energy_threshold,
                                      num_regions=num_regions_to_detect)

        use_demo = request.form.get('use_demo', 'false') == 'true'
        file_path = None # Initialize file_path

        # --- Load Data ---
        if use_demo:
            update_processing_status('loading', 5, 'Generating synthetic demo data',
                                  console_log="Using synthetic demo data")
            if not estimator.load_data('demo'):
                 logger.error("Error creating synthetic data")
                 flash('Error creating synthetic data', 'error')
                 update_processing_status('error', 100, 'Failed to create synthetic data')
                 return redirect(url_for('index'))
            update_processing_status('loading', 10, 'Synthetic data generated successfully',
                                     console_log="Successfully created synthetic data")
            # Demo data specific settings
            measurement_points = [(150, 200), (300, 350)] # Example points for demo

        else:
            # --- Handle File Upload or Selection ---
            existing_file = request.form.get('existing_file', '')
            uploaded_file = request.files.get('file')

            if existing_file:
                filename = existing_file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if not os.path.exists(file_path) or not allowed_file(filename):
                    msg = f"Selected file '{filename}' not found or not allowed type."
                    logger.error(msg)
                    flash(msg, 'error')
                    update_processing_status('error', 100, msg)
                    return redirect(url_for('index'))
                update_processing_status('loading', 5, f'Using existing file: {filename}',
                                     console_log=f"Using existing file: {filename}")
                flash(f'Using existing file: {filename}', 'info')
            elif uploaded_file and uploaded_file.filename != '':
                filename = secure_filename(uploaded_file.filename)
                if not allowed_file(filename):
                     msg = f"File type not allowed: {filename}"
                     logger.error(msg)
                     flash(msg, 'error')
                     update_processing_status('error', 100, msg)
                     return redirect(url_for('index'))
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                update_processing_status('loading', 5, f'Saving uploaded file: {filename}',
                                      console_log=f"Saving uploaded file to {file_path}")
                try:
                    uploaded_file.save(file_path)
                    flash(f'File {filename} uploaded successfully', 'info')
                except Exception as e:
                     msg = f"Error saving uploaded file: {e}"
                     logger.error(msg)
                     flash(msg, 'error')
                     update_processing_status('error', 100, msg)
                     return redirect(url_for('index'))
            else:
                logger.error("No file selected or uploaded")
                flash('No file selected or uploaded', 'error')
                update_processing_status('error', 100, 'No file provided')
                return redirect(url_for('index'))

            # --- Load Data from File ---
            update_processing_status('loading', 10, f'Loading data file: {filename}',
                                  console_log=f"Loading data file: {file_path}")
            # Add progress callback if load_data supports it, otherwise remove
            # def progress_callback(step, message): ...
            if not estimator.load_data(file_path):
                error_msg = f"Error loading data file: {filename}"
                logger.error(error_msg)
                if estimator.last_error:
                     error_msg += f" Details: {estimator.last_error.splitlines()[0]}" # Show first line of error
                flash(error_msg, 'error')
                update_processing_status('error', 100, error_msg)
                # Save error details if available
                if estimator.last_error:
                     with open(os.path.join(result_dir, 'error_loading.txt'), 'w') as f:
                          f.write(estimator.last_error)
                return redirect(url_for('index'))
            update_processing_status('processing', 25, f'{estimator.data_type.upper()} file loaded successfully',
                                      console_log=f"{estimator.data_type.upper()} file loaded successfully")

            # --- Save Raw Data Visualization ---
            if hasattr(estimator, 'raw_data_image') and estimator.raw_data_image is not None:
                # ... (save raw_data.png plot - existing code) ...
                plt.savefig(os.path.join(result_dir, 'raw_data.png'))
                plt.close(fig)
                update_processing_status('processing', 25, 'Raw data visualization saved',
                                      console_log="Saved raw data visualization")

            # --- Initialize Measurement Points (if not demo) ---
            # Choose default measurement points based on the loaded data dimensions
            img_shape = None
            if estimator.data_type == 'cphd' and estimator.raw_data is not None:
                 img_shape = estimator.raw_data.shape
            elif estimator.data_type == 'sicd' and estimator.raw_complex_data is not None:
                 img_shape = estimator.raw_complex_data.shape

            if img_shape:
                 height, width = img_shape # Assuming (azimuth, range) or (row, col)
                 # Define points relative to image size
                 measurement_points = [(height // 4, width // 4), (height * 3 // 4, width * 3 // 4)]
                 append_to_detailed_log(f"Defined default measurement points based on image size {img_shape}: {measurement_points}")
            else:
                 measurement_points = [(100, 100), (200, 200)] # Default fallback
                 append_to_detailed_log(f"Could not determine image size, using fallback measurement points: {measurement_points}")


        # --- Processing Steps (Conditional based on data type) ---

        # 1. Create Subapertures (Applies to both CPHD and SICD via internal logic)
        update_processing_status('processing', 30, 'Creating subapertures/sub-images...',
                              console_log="Creating subapertures/sub-images")
        if not estimator.create_subapertures():
            error_msg = "Error creating subapertures/sub-images"
            logger.error(error_msg)
            if estimator.last_error: error_msg += f": {estimator.last_error.splitlines()[0]}"
            flash(error_msg, 'error')
            update_processing_status('error', 100, error_msg)
            # Save error details
            if estimator.last_error:
                 with open(os.path.join(result_dir, 'error_subaperture.txt'), 'w') as f: f.write(estimator.last_error)
            return redirect(url_for('index'))
        # Save subaperture visualization (optional, might need complex data handling)
        # ... (code to plot subaperture spectrum - potentially adapt for complex SICD subimages) ...
        update_processing_status('processing', 35, 'Subapertures/sub-images created successfully',
                                  console_log=f"Created {len(estimator.subapertures)} subapertures/sub-images")


        # 2. Focus Subapertures (CPHD) / Prepare Magnitude Images (SICD)
        update_processing_status('processing', 40, 'Focusing (CPHD) / Preparing magnitude images (SICD)...',
                              console_log="Focusing (CPHD) / Preparing magnitude images (SICD)")
        if not estimator.focus_subapertures():
            error_msg = "Error during focusing/magnitude preparation"
            logger.error(error_msg)
            if estimator.last_error: error_msg += f": {estimator.last_error.splitlines()[0]}"
            flash(error_msg, 'error')
            update_processing_status('error', 100, error_msg)
             # Save error details
            if estimator.last_error:
                 with open(os.path.join(result_dir, 'error_focusing.txt'), 'w') as f: f.write(estimator.last_error)
            return redirect(url_for('index'))
        # Save SLC/magnitude images visualization
        if hasattr(estimator, 'slc_images') and estimator.slc_images:
            # ... (save slc_images.png plot - existing code should work) ...
            plt.savefig(os.path.join(result_dir, 'slc_images.png'))
            plt.close(fig)
            update_processing_status('processing', 45, 'Focusing/preparation complete',
                                      console_log=f"Generated {len(estimator.slc_images)} SLC/magnitude images")


        # 3. Estimate Displacement (Common step, uses magnitude images)
        # ... (Monitor memory usage before) ...
        update_processing_status('processing', 50, 'Estimating displacement...',
                             console_log="Estimating displacement using enhanced memory-efficient algorithm")
        try:
            # ... (Call estimate_displacement_enhanced - existing code) ...
            result = estimator.estimate_displacement_enhanced()
            # ... (Monitor memory usage after) ...
            if not result:
                # ... (Handle displacement estimation error - existing code) ...
                # Save error details
                if estimator.last_error:
                     with open(os.path.join(result_dir, 'error_displacement.txt'), 'w') as f: f.write(estimator.last_error)
                return redirect(url_for('index'))
        except Exception as e:
            # ... (Handle unexpected displacement error - existing code) ...
            return redirect(url_for('index'))
        # Save displacement maps visualization
        if hasattr(estimator, 'displacement_maps') and estimator.displacement_maps:
            # ... (save displacement_maps.png plot - existing code) ...
             plt.savefig(os.path.join(result_dir, 'displacement_maps.png'))
             plt.close(fig)
             update_processing_status('processing', 60, 'Displacement estimated successfully',
                                   console_log="Displacement maps calculated successfully")


        # 4. Analyze Time Series (Common step)
        update_processing_status('processing', 90, 'Analyzing time series...',
                              console_log=f"Analyzing time series at {len(measurement_points)} measurement points")
        if not hasattr(estimator, 'displacement_maps') or not estimator.displacement_maps:
             # ... (Handle missing displacement maps error - existing code) ...
             return redirect(url_for('index'))
        if not estimator.analyze_time_series(measurement_points):
            # ... (Handle time series analysis error - existing code) ...
            return redirect(url_for('index'))
        # Save individual measurement point results
        for i in range(len(measurement_points)):
             # ... (save point_i_results.png plot - existing code) ...
             estimator.plot_results(i, output_file=point_result_file)


        # 5. Calculate Vibration Energy & Detect Regions (Common steps)
        update_processing_status('saving', 92, 'Calculating vibration energy & detecting regions...',
                              console_log="Calculating vibration energy map and detecting regions")
        try:
            if estimator.calculate_vibration_energy_map():
                # ... (Detect regions using detector instance - existing code) ...
                detected_regions = detector.detect_regions(estimator.vibration_energy_map_db)
                estimator.ship_regions = detected_regions
                # ... (Plot regions or energy map - existing code) ...
                if detected_regions:
                     # ... plot regions ...
                     ship_regions_file = os.path.join(result_dir, 'ship_regions.png')
                     if estimator.plot_ship_regions(output_file=ship_regions_file):
                          append_to_detailed_log(f"Saved ship regions visualization to {ship_regions_file}")
                else:
                     # ... plot energy map only ...
                     energy_map_file = os.path.join(result_dir, 'vibration_energy_map_only.png')
                     if estimator.plot_vibration_energy_map(output_file=energy_map_file):
                          append_to_detailed_log(f"Saved vibration energy map (no regions detected) to {energy_map_file}")

            else:
                # ... (Handle failure to calculate energy map - existing code) ...
                append_to_detailed_log("Failed to calculate vibration energy map. Skipping region detection.")
        except Exception as e:
            # ... (Handle errors in energy/detection steps - existing code) ...
            append_to_detailed_log(f"Error processing vibration energy/regions: {str(e)}")


        # --- Finalize: Save Metadata, Summary, Complete Status ---
        update_processing_status('saving', 95, 'Saving final results and metadata...',
                              console_log=f"Saving results to {result_dir}")

        metadata = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                # ... (estimator params) ...
                'num_subapertures': num_subapertures,
                'window_size': window_size,
                'overlap': overlap,
                # ... (detector params) ...
                'min_region_size': min_region_size,
                'energy_threshold': energy_threshold,
                'num_regions_to_detect': num_regions_to_detect
            },
            'file': os.path.basename(file_path) if file_path else 'demo',
            'data_type': estimator.data_type, # Add data type used
            'measurement_points': [list(mp) for mp in measurement_points],
            'detected_regions_count': len(estimator.ship_regions) if hasattr(estimator, 'ship_regions') and estimator.ship_regions else 0,
            # 'processing_time': ... # Calculate actual duration
        }
        # ... (Save metadata.json - existing code) ...
        with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # ... (Save summary.txt - existing code, maybe add data_type) ...
        with open(os.path.join(result_dir, 'summary.txt'), 'w') as f:
             f.write(f"Processing completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
             f.write(f"Input file: {os.path.basename(file_path) if file_path else 'demo'} (Type: {estimator.data_type.upper()})\n") # Added type
             # ... (rest of summary) ...

        update_processing_status('complete', 100,
                              f'Processing completed. Results saved to {os.path.basename(result_dir)}',
                              console_log="Processing completed successfully")
        flash(f'Processing completed successfully. Results saved to {os.path.basename(result_dir)}', 'success')

    except Exception as e:
        # ... (General error handling - existing code) ...
        logger.exception(f"Unexpected error during processing: {str(e)}")
        # ... (save traceback, flash message, update status) ...

    finally:
        # ... (Remove file handler, write final log status - existing code) ...
        logger.removeHandler(file_handler)
        # ...

    return redirect(url_for('index'))

# ... (results_file, main block remain the same) ...

```

## Usage

1.  Ensure the `sarpy` library is installed (`pip install sarpy`).
2.  Upload either a CPHD file (e.g., `*.cphd`) or a SICD file (e.g., `*.ntf`, `*.nitf`) through the web interface.
3.  The application will automatically detect the type based on the file extension and internal checks during loading.
4.  The processing pipeline will adjust accordingly, notably skipping the focusing step for SICD data.
5.  Results (displacement maps, time series, vibration energy, detected regions) will be generated based on the input data type.

## Future Considerations

*   **SICD Timing:** The accuracy of frequency analysis for SICD depends heavily on correctly extracting timing information (`start_time`, `duration`) from the SICD metadata. The current implementation (`_get_sicd_timing_info`) includes basic attempts but may need refinement based on the specific SICD files used.
*   **Complex Correlation:** The current pipeline uses magnitude images for displacement estimation for both CPHD and SICD. For potentially higher accuracy, especially with complex SICD data, implementing complex cross-correlation in the `estimate_displacement_enhanced` step could be explored.
*   **UI Selection:** Allow the user to explicitly select the data type via a dropdown, overriding automatic detection if needed.
*   **Error Handling:** Enhance error handling for `sarpy` loading issues specific to CPHD vs. SICD.
*   **Demo Data:** Update the `create_synthetic_data` function to optionally generate SICD-like complex image data for testing the SICD path. The current implementation assumes the demo generates complex CPHD-like sub-apertures.

</rewritten_file> 