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

def split_aperture(cphd_data, num_subapertures=7):
    """
    Split the phase history data into multiple Doppler sub-apertures
    
    Parameters
    ----------
    cphd_data : numpy.ndarray
        Complex phase history data
    num_subapertures : int, optional
        Number of sub-apertures to create (default is 7)
        
    Returns
    -------
    list
        List of numpy arrays containing the sub-aperture data
    """
    # Get the dimensions of the data
    num_pulses, num_samples = cphd_data.shape
    
    # Calculate the size of each sub-aperture
    subaperture_size = num_pulses // num_subapertures
    
    # Create the sub-apertures
    subapertures = []
    for i in range(num_subapertures):
        start_idx = i * subaperture_size
        end_idx = start_idx + subaperture_size
        
        # Handle the last sub-aperture which might be smaller
        if i == num_subapertures - 1:
            end_idx = num_pulses
            
        subaperture = cphd_data[start_idx:end_idx, :]
        subapertures.append(subaperture)
    
    return subapertures

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
        subapertures = split_aperture(data)
        print(f"\nCreated {len(subapertures)} sub-apertures")
        for i, subap in enumerate(subapertures):
            print(f"Sub-aperture {i+1} shape: {subap.shape}")
            
    except Exception as e:
        print(f"Error: {e}")
