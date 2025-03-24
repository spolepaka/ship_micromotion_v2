import sarpy
from importlib import import_module

# Print SarPy version
print('SarPy version:', sarpy.__version__)

# Explore CPHD module
try:
    cphd = import_module('sarpy.io.phase_history.cphd')
    print('\nCPHD Module available classes:')
    for item in dir(cphd):
        if not item.startswith('__'):
            print(f'- {item}')
except ImportError as e:
    print(f'Error importing CPHD module: {e}')

# Explore IO module
try:
    io = import_module('sarpy.io')
    print('\nIO Module available functions:')
    for item in dir(io):
        if not item.startswith('__'):
            print(f'- {item}')
except ImportError as e:
    print(f'Error importing IO module: {e}')

# Explore complex module
try:
    complex_module = import_module('sarpy.io.complex')
    print('\nComplex Module available functions:')
    for item in dir(complex_module):
        if not item.startswith('__'):
            print(f'- {item}')
except ImportError as e:
    print(f'Error importing Complex module: {e}')

# Explore processing module
try:
    processing = import_module('sarpy.processing')
    print('\nProcessing Module available functions:')
    for item in dir(processing):
        if not item.startswith('__'):
            print(f'- {item}')
except ImportError as e:
    print(f'Error importing Processing module: {e}')
