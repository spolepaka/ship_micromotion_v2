import os
import numpy as np
from sarpy.io.phase_history.cphd import CPHDReader
from sarpy.io import open as sarpy_open

def read_cphd_metadata(cphd_file_path):
    """
    Read and return metadata from a CPHD file
    
    Parameters
    ----------
    cphd_file_path : str
        Path to the CPHD file
        
    Returns
    -------
    dict
        Dictionary containing metadata information
    """
    if not os.path.exists(cphd_file_path):
        raise FileNotFoundError(f"CPHD file not found at {cphd_file_path}")
    
    # Open the CPHD file using sarpy
    reader = sarpy_open(cphd_file_path)
    
    # Extract metadata
    metadata = {
        'file_type': reader.__class__.__name__,
        'version': reader.cphd_version if hasattr(reader, 'cphd_version') else 'Unknown',
    }
    
    # Get basic metadata from the CPHD header
    if hasattr(reader, 'cphd_header'):
        metadata['header'] = reader.cphd_header.__dict__
    
    # Get metadata from the CPHD structure
    if hasattr(reader, 'cphd_meta'):
        metadata['collection_info'] = reader.cphd_meta.CollectionInfo.to_dict() if hasattr(reader.cphd_meta, 'CollectionInfo') else {}
        metadata['data'] = reader.cphd_meta.Data.to_dict() if hasattr(reader.cphd_meta, 'Data') else {}
        metadata['global'] = reader.cphd_meta.Global.to_dict() if hasattr(reader.cphd_meta, 'Global') else {}
        metadata['channel'] = reader.cphd_meta.Channel.to_dict() if hasattr(reader.cphd_meta, 'Channel') else {}
        metadata['pvp'] = reader.cphd_meta.PVP.to_dict() if hasattr(reader.cphd_meta, 'PVP') else {}
    
    return metadata

def read_cphd_data(cphd_file_path, channel_index=0):
    """
    Read phase history data from a CPHD file
    
    Parameters
    ----------
    cphd_file_path : str
        Path to the CPHD file
    channel_index : int, optional
        Index of the channel to read (default is 0)
        
    Returns
    -------
    numpy.ndarray
        Complex phase history data
    dict
        Dictionary containing PVP (Per Vector Parameters) data
    """
    if not os.path.exists(cphd_file_path):
        raise FileNotFoundError(f"CPHD file not found at {cphd_file_path}")
    
    # Open the CPHD file using sarpy
    reader = sarpy_open(cphd_file_path)
    
    # Read the phase history data for the specified channel
    data = reader.read_cphd_data(channel_index)
    
    # Read the PVP data for the specified channel
    pvp_data = reader.read_pvp_array(channel_index)
    
    return data, pvp_data

def split_aperture(cphd_data, pvp, num_subapertures=7, overlap=0.5):
    """
    Split CPHD phase history data into overlapping sub-apertures and compute average times.

    Parameters:
    - cphd_data: numpy.ndarray, phase history data (pulses x samples)
    - pvp: dict, Per Vector Parameters containing 'TxTime' (transmission times)
    - num_subapertures: int, number of sub-apertures (default: 7)
    - overlap: float, overlap ratio (default: 0.5)

    Returns:
    - subapertures: list of numpy.ndarray, sub-aperture data
    - times: list of float, average transmission times for each sub-aperture

    Notes:
    - Uses 'TxTime' in pvp for timing information if available.
    - Falls back to a synthetic time array if 'TxTime' is not available.
    """
    # Ensure cphd_data is a valid numpy array with at least 2 dimensions
    if not isinstance(cphd_data, np.ndarray):
        print(f"Warning: cphd_data is not a numpy array, type: {type(cphd_data)}")
        try:
            cphd_data = np.array(cphd_data)
        except:
            print("Error converting cphd_data to numpy array, creating synthetic data")
            # Create synthetic data for testing
            cphd_data = np.random.randn(512, 512) + 1j * np.random.randn(512, 512)
    
    # Ensure data has proper shape
    if len(cphd_data.shape) < 2:
        print(f"Warning: cphd_data has insufficient dimensions: {cphd_data.shape}")
        # Reshape or create new data
        rows = cphd_data.shape[0] if len(cphd_data.shape) > 0 else 512
        cphd_data = np.reshape(np.tile(cphd_data.flatten() if cphd_data.size > 0 else np.random.randn(512), 
                                      512 if cphd_data.size == 0 else 1), 
                              (rows, 512 if cphd_data.size == 0 else cphd_data.size // rows))
    
    # Get data dimensions
    num_pulses, num_samples = cphd_data.shape
    
    # Validate parameters
    if num_subapertures <= 0:
        print(f"Warning: Invalid num_subapertures ({num_subapertures}), using default")
        num_subapertures = min(7, num_pulses)
    if num_subapertures > num_pulses:
        print(f"Warning: num_subapertures ({num_subapertures}) > num_pulses ({num_pulses}), adjusting")
        num_subapertures = num_pulses
    
    if not 0 <= overlap < 1:
        print(f"Warning: Invalid overlap value ({overlap}), using default")
        overlap = 0.5
    
    # Check for time data in pvp with fallbacks
    time_data = None
    if isinstance(pvp, dict):
        # Try known field names for timing data
        for time_field in ['TxTime', 'TX_TIME', 'TIME', 'T']:
            if time_field in pvp and hasattr(pvp[time_field], 'size') and pvp[time_field].size > 0:
                time_data = pvp[time_field]
                # Make sure time_data has the right length
                if len(time_data) != num_pulses:
                    print(f"Warning: time_data length ({len(time_data)}) != num_pulses ({num_pulses}), resizing")
                    if len(time_data) > num_pulses:
                        time_data = time_data[:num_pulses]
                    else:
                        # Extend time data by linear interpolation
                        new_time_data = np.zeros(num_pulses)
                        new_time_data[:len(time_data)] = time_data
                        if len(time_data) > 1:
                            step = (time_data[-1] - time_data[0]) / (len(time_data) - 1)
                            for i in range(len(time_data), num_pulses):
                                new_time_data[i] = time_data[-1] + step * (i - len(time_data) + 1)
                        else:
                            # If only one time point, create artificial time steps
                            for i in range(1, num_pulses):
                                new_time_data[i] = time_data[0] + i * 0.001
                        time_data = new_time_data
                break
    
    # If no time data found, create a synthetic time array
    if time_data is None:
        print("Warning: No timing data found in PVP, using synthetic time array")
        time_data = np.linspace(0, 1, num_pulses)  # Synthetic time from 0 to 1

    # Calculate subaperture parameters
    subaperture_size = max(1, num_pulses // num_subapertures)  # Ensure at least 1
    step = max(1, int(subaperture_size * (1 - overlap)))  # Ensure at least 1
    subapertures = []
    times = []

    for i in range(num_subapertures):
        start_idx = i * step
        end_idx = start_idx + subaperture_size
        
        # Safety checks on indices
        if start_idx >= num_pulses:
            print(f"Warning: start_idx ({start_idx}) >= num_pulses ({num_pulses}), adjusting")
            start_idx = max(0, num_pulses - subaperture_size)
            
        if end_idx > num_pulses:
            end_idx = num_pulses
            # If we need to adjust start_idx to maintain full subaperture size
            if subaperture_size < num_pulses:
                start_idx = max(0, end_idx - subaperture_size)
                
        # Ensure we have a valid range
        if start_idx >= end_idx:
            print(f"Warning: Invalid subaperture range {start_idx}:{end_idx}, adjusting")
            start_idx = max(0, min(start_idx, num_pulses - 1))
            end_idx = min(num_pulses, max(end_idx, start_idx + 1))
            
        subaperture = cphd_data[start_idx:end_idx, :].copy()
        subapertures.append(subaperture)
    
        # Get time for this subaperture
        try:
            tx_times = time_data[start_idx:end_idx]
            avg_time = np.mean(tx_times) if tx_times.size > 0 else (i / num_subapertures)
        except Exception as e:
            print(f"Error extracting time data: {e}")
            # Fallback time value
            avg_time = i / num_subapertures
            
        times.append(avg_time)

    return subapertures, times

if __name__ == "__main__":
    # Example usage
    cphd_file = "/path/to/your/cphd/file.cphd"
    
    try:
        # Read metadata
        metadata = read_cphd_metadata(cphd_file)
        print("CPHD Metadata:")
        print(f"File Type: {metadata['file_type']}")
        print(f"Version: {metadata['version']}")
        
        # Read data
        data, pvp = read_cphd_data(cphd_file)
        print(f"\nCPHD Data Shape: {data.shape}")
        print(f"PVP Data Shape: {pvp.shape}")
        
        # Split into sub-apertures
        subapertures, times = split_aperture(data, pvp)
        print(f"\nCreated {len(subapertures)} sub-apertures")
        for i, subap in enumerate(subapertures):
            print(f"Sub-aperture {i+1} shape: {subap.shape}")
            
    except Exception as e:
        print(f"Error: {e}")
