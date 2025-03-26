#!/usr/bin/env python
import os
import sys
from sarpy.io import open as sarpy_open

def analyze_cphd(file_path):
    """Analyze CPHD file structure to determine how to correctly load it."""
    print(f"Analyzing CPHD file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    try:
        # Try to open with sarpy's open function
        reader = sarpy_open(file_path)
        print(f"\nReader type: {type(reader).__name__}")
        print(f"Reader class hierarchy: {type(reader).__mro__}")
        
        # Print available attributes and methods
        methods = [m for m in dir(reader) if not m.startswith('_')]
        print(f"\nAvailable methods ({len(methods)}):")
        for method in sorted(methods[:20]):
            print(f"  - {method}")
        if len(methods) > 20:
            print(f"  ... and {len(methods) - 20} more methods")
        
        # Check if it's a CPHD reader
        if hasattr(reader, 'cphd_meta'):
            print("\nCPHD metadata available!")
            meta = reader.cphd_meta
            print(f"CPHD version: {reader.cphd_version if hasattr(reader, 'cphd_version') else 'Unknown'}")
            
            # Check for Data section
            if hasattr(meta, 'Data'):
                print("\nData section:")
                data_attrs = dir(meta.Data)
                data_dict = meta.Data.to_dict() if hasattr(meta.Data, 'to_dict') else {}
                for key, value in data_dict.items():
                    print(f"  - {key}: {value}")
            
            # Check for channels
            if hasattr(meta, 'Channel'):
                print("\nChannel information:")
                chan_info = meta.Channel.to_dict() if hasattr(meta.Channel, 'to_dict') else {}
                print(f"  Channels: {chan_info}")
        
        # Try to read data
        if hasattr(reader, 'read_chip'):
            print("\nReader has read_chip method. Trying to get dimensions...")
            try:
                if hasattr(reader, 'data_size'):
                    print(f"  Data size: {reader.data_size}")
                    
                # Try to read a small section
                if hasattr(reader, 'read_chip'):
                    print("  Attempting to read a small chip...")
                    chip = reader.read_chip()
                    print(f"  Successfully read chip of shape: {chip.shape}")
            except Exception as e:
                print(f"  Error reading chip: {str(e)}")
        
        # Try to read PVP data
        pvp_methods = [m for m in methods if 'pvp' in m.lower()]
        if pvp_methods:
            print("\nPVP-related methods:")
            for method in pvp_methods:
                print(f"  - {method}")
            
            try:
                if 'read_pvp_array' in pvp_methods:
                    print("  Attempting to read PVP array...")
                    pvp = reader.read_pvp_array() if hasattr(reader, 'read_pvp_array') else None
                    if pvp is not None:
                        print(f"  PVP data keys: {list(pvp.keys()) if isinstance(pvp, dict) else 'Not a dictionary'}")
                        if isinstance(pvp, dict) and 'TxTime' in pvp:
                            print(f"  TxTime shape: {pvp['TxTime'].shape}")
                
                elif 'read_pvp' in pvp_methods:
                    print("  Attempting to read PVP...")
                    pvp = reader.read_pvp() if hasattr(reader, 'read_pvp') else None
                    if pvp is not None:
                        print(f"  PVP data keys: {list(pvp.keys()) if isinstance(pvp, dict) else 'Not a dictionary'}")
            except Exception as e:
                print(f"  Error reading PVP: {str(e)}")
        
        return True
    
    except Exception as e:
        print(f"Error analyzing CPHD file: {str(e)}")
        traceback_info = sys.exc_info()[2]
        print(f"Error location: {traceback_info.tb_frame.f_code.co_filename}, line {traceback_info.tb_lineno}")
        return False

if __name__ == "__main__":
    file_path = 'uploads/2023-08-31-01-09-38_UMBRA-04_CPHD.cphd'
    analyze_cphd(file_path) 