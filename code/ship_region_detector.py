import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, binary_erosion, binary_dilation

class ShipRegionDetector:
    """
    Class for detecting and labeling ship regions in SAR images based on vibration energy
    """
    
    def __init__(self, min_region_size=10, energy_threshold=-15, num_regions=3):
        """
        Initialize the ship region detector
        
        Parameters
        ----------
        min_region_size : int, optional
            Minimum size of a region in pixels (default is 10)
        energy_threshold : float, optional
            Energy threshold in dB for ship detection (default is -15)
        num_regions : int, optional
            Number of regions to detect (default is 3)
        """
        self.min_region_size = min_region_size
        self.energy_threshold = energy_threshold
        self.num_regions = num_regions
        self.ship_regions = None
        
    def detect_regions(self, vibration_energy_map_db):
        """
        Detect ship regions based on vibration energy
        
        Parameters
        ----------
        vibration_energy_map_db : numpy.ndarray
            Vibration energy map in dB scale
            
        Returns
        -------
        list
            List of dictionaries containing region information
        """
        # Threshold the energy map to find ship pixels
        ship_mask = vibration_energy_map_db > self.energy_threshold
        
        # If no ship pixels found, return empty list
        if not np.any(ship_mask):
            print("Warning: No ship regions detected")
            return []
        
        # Apply morphological operations to clean up the mask
        ship_mask = binary_erosion(ship_mask, disk(1))
        ship_mask = binary_dilation(ship_mask, disk(2))
        
        # Find connected components
        labeled_array, num_features = ndimage.label(ship_mask)
        
        # If no regions found, return empty list
        if num_features == 0:
            print("Warning: No connected regions found")
            return []
        
        # Measure properties of labeled regions
        region_props = measure.regionprops(labeled_array)
        
        # Filter regions by size
        valid_regions = []
        for prop in region_props:
            if prop.area >= self.min_region_size:
                valid_regions.append(prop)
        
        # If no valid regions found, return empty list
        if len(valid_regions) == 0:
            print(f"Warning: No regions larger than {self.min_region_size} pixels found")
            return []
        
        # Sort regions by vibration energy (sum of energy in region)
        region_energies = []
        for prop in valid_regions:
            region_mask = labeled_array == prop.label
            region_energy = np.sum(vibration_energy_map_db[region_mask])
            region_energies.append((prop, region_energy))
        
        # Sort by energy (descending)
        region_energies.sort(key=lambda x: x[1], reverse=True)
        
        # Select top regions
        num_to_select = min(self.num_regions, len(region_energies))
        selected_regions = region_energies[:num_to_select]
        
        # Create region information
        ship_regions = []
        for i, (prop, energy) in enumerate(selected_regions):
            # Create region mask
            region_mask = labeled_array == prop.label
            
            # Get region centroid
            centroid_r, centroid_c = prop.centroid
            
            # Store region info
            ship_regions.append({
                'id': i + 1,  # 1-based indexing as in the example
                'mask': region_mask,
                'centroid': (centroid_r, centroid_c),
                'energy': energy,
                'area': prop.area,
                'bbox': prop.bbox  # (min_row, min_col, max_row, max_col)
            })
        
        self.ship_regions = ship_regions
        return ship_regions
    
    def segment_regions_watershed(self, vibration_energy_map_db):
        """
        Segment ship regions using watershed algorithm for more precise boundaries
        
        Parameters
        ----------
        vibration_energy_map_db : numpy.ndarray
            Vibration energy map in dB scale
            
        Returns
        -------
        list
            List of dictionaries containing region information
        """
        # Threshold the energy map to find ship pixels
        ship_mask = vibration_energy_map_db > self.energy_threshold
        
        # If no ship pixels found, return empty list
        if not np.any(ship_mask):
            print("Warning: No ship regions detected")
            return []
        
        # Apply morphological operations to clean up the mask
        ship_mask = binary_erosion(ship_mask, disk(1))
        ship_mask = binary_dilation(ship_mask, disk(2))
        
        # Find local maxima as markers for watershed
        distance = ndimage.distance_transform_edt(ship_mask)
        local_max = peak_local_max(distance, indices=False, min_distance=20, labels=ship_mask)
        markers = ndimage.label(local_max)[0]
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=ship_mask)
        
        # Measure properties of labeled regions
        region_props = measure.regionprops(labels)
        
        # Filter regions by size
        valid_regions = []
        for prop in region_props:
            if prop.area >= self.min_region_size:
                valid_regions.append(prop)
        
        # If no valid regions found, return empty list
        if len(valid_regions) == 0:
            print(f"Warning: No regions larger than {self.min_region_size} pixels found")
            return []
        
        # Sort regions by vibration energy (sum of energy in region)
        region_energies = []
        for prop in valid_regions:
            region_mask = labels == prop.label
            region_energy = np.sum(vibration_energy_map_db[region_mask])
            region_energies.append((prop, region_energy))
        
        # Sort by energy (descending)
        region_energies.sort(key=lambda x: x[1], reverse=True)
        
        # Select top regions
        num_to_select = min(self.num_regions, len(region_energies))
        selected_regions = region_energies[:num_to_select]
        
        # Create region information
        ship_regions = []
        for i, (prop, energy) in enumerate(selected_regions):
            # Create region mask
            region_mask = labels == prop.label
            
            # Get region centroid
            centroid_r, centroid_c = prop.centroid
            
            # Store region info
            ship_regions.append({
                'id': i + 1,  # 1-based indexing as in the example
                'mask': region_mask,
                'centroid': (centroid_r, centroid_c),
                'energy': energy,
                'area': prop.area,
                'bbox': prop.bbox  # (min_row, min_col, max_row, max_col)
            })
        
        self.ship_regions = ship_regions
        return ship_regions
    
    def get_region_statistics(self, vibration_energy_map_db, displacement_maps=None):
        """
        Calculate statistics for each detected region
        
        Parameters
        ----------
        vibration_energy_map_db : numpy.ndarray
            Vibration energy map in dB scale
        displacement_maps : list, optional
            List of displacement maps for calculating additional statistics
            
        Returns
        -------
        list
            List of dictionaries containing region statistics
        """
        if self.ship_regions is None:
            print("Error: No ship regions detected")
            return []
        
        region_stats = []
        for region in self.ship_regions:
            # Get region mask
            mask = region['mask']
            
            # Calculate energy statistics
            energy_values = vibration_energy_map_db[mask]
            stats = {
                'id': region['id'],
                'centroid': region['centroid'],
                'area': region['area'],
                'mean_energy': np.mean(energy_values),
                'max_energy': np.max(energy_values),
                'min_energy': np.min(energy_values),
                'std_energy': np.std(energy_values)
            }
            
            # Calculate displacement statistics if available
            if displacement_maps is not None:
                range_displacements = []
                azimuth_displacements = []
                
                for range_map, azimuth_map in displacement_maps:
                    # Downsample mask to match displacement map size if needed
                    if mask.shape != range_map.shape:
                        # Simple downsampling by averaging
                        factor_r = mask.shape[0] / range_map.shape[0]
                        factor_c = mask.shape[1] / range_map.shape[1]
                        
                        downsampled_mask = np.zeros(range_map.shape, dtype=bool)
                        for r in range(range_map.shape[0]):
                            for c in range(range_map.shape[1]):
                                r_start = int(r * factor_r)
                                r_end = int((r + 1) * factor_r)
                                c_start = int(c * factor_c)
                                c_end = int((c + 1) * factor_c)
                                
                                # If any pixel in the original mask is True, set downsampled pixel to True
                                if np.any(mask[r_start:r_end, c_start:c_end]):
                                    downsampled_mask[r, c] = True
                        
                        mask_for_disp = downsampled_mask
                    else:
                        mask_for_disp = mask
                    
                    # Extract displacements for the region
                    range_values = range_map[mask_for_disp]
                    azimuth_values = azimuth_map[mask_for_disp]
                    
                    range_displacements.extend(range_values)
                    azimuth_displacements.extend(azimuth_values)
                
                # Calculate displacement statistics
                stats.update({
                    'mean_range_disp': np.mean(range_displacements) if range_displacements else 0,
                    'std_range_disp': np.std(range_displacements) if range_displacements else 0,
                    'mean_azimuth_disp': np.mean(azimuth_displacements) if azimuth_displacements else 0,
                    'std_azimuth_disp': np.std(azimuth_displacements) if azimuth_displacements else 0
                })
            
            region_stats.append(stats)
        
        return region_stats
