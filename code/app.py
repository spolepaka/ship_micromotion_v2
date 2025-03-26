from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import datetime
import logging
import json
from werkzeug.utils import secure_filename
from micromotion_estimator import ShipMicroMotionEstimator
from ship_region_detector import ShipRegionDetector
from test_estimator import create_synthetic_data
from flask_session import Session  # Import Flask-Session
import time
import psutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('ship_micromotion')

# Filter out matplotlib font_manager debug logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
# Additional matplotlib modules that might be verbose
logging.getLogger('matplotlib.backends').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)

app = Flask(__name__)
app.secret_key = 'ship_micromotion_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.getcwd(), 'results')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5 GB max upload size
app.config['SESSION_TYPE'] = 'filesystem'  # For larger session data
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')

# Initialize the session extension
Session(app)

# Create upload and results folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  # Create session directory

# Allowed file extensions
ALLOWED_EXTENSIONS = {'cphd', 'h5', 'hdf5'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_uploaded_files():
    """List all files in the uploads directory that have allowed extensions"""
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            files.append(filename)
    return files

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

# Create a timestamped folder for results
def create_results_folder():
    """Create a new results folder with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], f"results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Created results directory: {result_dir}")
    return result_dir

# Function to update processing status
def update_processing_status(status, progress=None, details=None, console_log=None):
    """Update the processing status in the session"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # If console_log is provided, log it to console and file
    if console_log:
        logger.info(console_log)
    
    # Get current status to preserve logs
    current_status = session.get('processing_status', {})
    log_messages = current_status.get('log_messages', [])
    
    session['processing_status'] = {
        'status': status,
        'progress': progress,
        'details': details,
        'timestamp': timestamp,
        'log_messages': log_messages
    }
    
    # Add the new message to the log history (keep last 15 messages)
    if details:
        log_entry = f"{timestamp}: {details}"
        session['processing_status']['log_messages'].append(log_entry)
        # Keep only the most recent 15 messages
        session['processing_status']['log_messages'] = session['processing_status']['log_messages'][-15:]
    
    session.modified = True
    
    # Immediately commit the session - this is important to ensure updates are visible
    if hasattr(session, 'save_session'):
        session.save_session(None)  # Force session save
    
    logger.debug(f"Updated processing status: {status}, progress: {progress}, details: {details}")

# Add a function to copy console output to the processing log
def log_to_processing(message):
    """Log a message to both console and processing log"""
    logger.info(message)
    return message

# Steps of the algorithm for debugging
ALGORITHM_STEPS = [
    'load_data',
    'create_subapertures',
    'focus_subapertures',
    'estimate_displacement',
    'analyze_time_series',
    'calculate_vibration_energy',
    'detect_ship_regions'
]

# Definition of expected processing times for each step
STEP_DURATIONS = {
    'load_data': 10,            # 10 seconds
    'create_subapertures': 5,   # 5 seconds
    'focus_subapertures': 15,   # 15 seconds
    'estimate_displacement': 20, # 20 seconds
    'analyze_time_series': 8,    # 8 seconds
    'calculate_vibration_energy': 12, # 12 seconds
    'detect_ship_regions': 5     # 5 seconds
}

@app.route('/')
def index():
    uploaded_files = list_uploaded_files()
    # Initialize processing status if not exists
    if 'processing_status' not in session:
        session['processing_status'] = {
            'status': 'idle',
            'progress': None,
            'details': None,
            'timestamp': None
        }
        
    # Get recent result folders
    results_folders = []
    if os.path.exists(app.config['RESULTS_FOLDER']):
        try:
            # Get all result directories
            all_results = [d for d in os.listdir(app.config['RESULTS_FOLDER']) 
                           if os.path.isdir(os.path.join(app.config['RESULTS_FOLDER'], d)) and d.startswith('results_')]
            
            # Sort by creation time (newest first)
            all_results.sort(key=lambda x: os.path.getctime(os.path.join(app.config['RESULTS_FOLDER'], x)), reverse=True)
            
            # Take only the 5 most recent
            recent_results = all_results[:5]
            
            for result_dir in recent_results:
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_dir)
                
                # Check for metadata file
                metadata_file = os.path.join(result_path, 'metadata.json')
                metadata = None
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                # Get list of result files
                result_files = []
                for f in os.listdir(result_path):
                    if f.endswith('.png'):
                        result_files.append({
                            'name': f,
                            'path': os.path.join('results', result_dir, f)
                        })
                
                # Create result info
                results_folders.append({
                    'name': result_dir,
                    'path': result_path,
                    'timestamp': metadata.get('timestamp', 'Unknown') if metadata else 'Unknown',
                    'file': metadata.get('file', 'Unknown') if metadata else 'Unknown',
                    'files': result_files,
                    'has_error': os.path.exists(os.path.join(result_path, 'error_details.txt'))
                })
        except Exception as e:
            logger.error(f"Error getting result folders: {e}")
    
    return render_template('index.html', 
                          uploaded_files=uploaded_files, 
                          processing_status=session.get('processing_status'),
                          results_folders=results_folders)

@app.route('/check_status')
def check_status():
    """API endpoint to check processing status"""
    return jsonify(session.get('processing_status', {'status': 'idle'}))

@app.route('/debug')
def debug_playground():
    """Render the debugging playground page"""
    uploaded_files = list_uploaded_files()
    return render_template('debug.html', 
                           uploaded_files=uploaded_files, 
                           algorithm_steps=ALGORITHM_STEPS)

@app.route('/run_debug_step', methods=['POST'])
def run_debug_step():
    """Run a specific algorithm step and return the results"""
    # Get parameters
    step = request.form.get('step')
    filename = request.form.get('filename')
    num_subapertures = int(request.form.get('num_subapertures', 7))
    window_size = int(request.form.get('window_size', 64))
    overlap = float(request.form.get('overlap', 0.5))
    use_demo = request.form.get('use_demo', 'false') == 'true'
    
    # Store parameters in session for persistence between steps
    session_data = {}
    session_data['current_step'] = step
    session_data['filename'] = filename
    session_data['num_subapertures'] = num_subapertures
    session_data['window_size'] = window_size
    session_data['overlap'] = overlap
    session_data['use_demo'] = use_demo
    
    # Create estimator
    estimator = ShipMicroMotionEstimator(
        num_subapertures=num_subapertures,
        window_size=window_size,
        overlap=overlap,
        debug_mode=True  # Enable debug mode
    )
    
    results = {}
    error = None
    
    # Process data loading
    if step == 'load_data':
        if use_demo:
            # Use synthetic data for demo
            try:
                raw_data, pvp, _ = create_synthetic_data(
                    rows=512, cols=512, num_subapertures=num_subapertures
                )
                
                # Set the raw data without processing subapertures
                estimator.raw_data = raw_data
                estimator.pvp = pvp
                estimator.raw_data_image = np.abs(raw_data)
                estimator.data_loaded = True
                
                results['message'] = 'Synthetic data loaded successfully'
                # Create a visualization of the raw data
                if hasattr(estimator, 'raw_data') and estimator.raw_data is not None:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    if hasattr(estimator, 'raw_data_image'):
                        ax.imshow(np.abs(estimator.raw_data_image), cmap='viridis', 
                                 aspect='auto', vmax=np.percentile(np.abs(estimator.raw_data_image), 95))
                        ax.set_title('Raw SAR Data (Amplitude)')
                        ax.set_xlabel('Range')
                        ax.set_ylabel('Azimuth')
                        plt.colorbar(ax.imshow(np.abs(estimator.raw_data_image), cmap='viridis', 
                                             aspect='auto', vmax=np.percentile(np.abs(estimator.raw_data_image), 95)),
                                     ax=ax, label='Amplitude')
                        results['raw_data_plot'] = fig_to_base64(fig)
                        plt.close(fig)
            except Exception as e:
                error = f'Error creating synthetic data: {str(e)}'
        else:
            # Use real data
            if not filename:
                error = 'No file selected'
            else:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if not os.path.exists(filepath):
                    error = f'File {filename} not found'
                else:
                    try:
                        from sarpy.io import open as sarpy_open
                        
                        print(f"Loading CPHD file: {filepath}")
                        # Use sarpy's generic open function which is more robust
                        reader = sarpy_open(filepath)
                        reader_type = type(reader).__name__
                        print(f"Reader type: {reader_type}")
                        
                        # Get metadata
                        if hasattr(reader, 'cphd_meta'):
                            metadata = reader.cphd_meta
                            print("CPHD metadata available")
                        else:
                            metadata = {}
                            print("Warning: No CPHD metadata found")
                        
                        # Read the data using read_chip which we know works based on our analysis
                        if hasattr(reader, 'read_chip'):
                            print("Reading CPHD data using read_chip...")
                            data = reader.read_chip()
                            print(f"Data shape: {data.shape}")
                        else:
                            raise ValueError("Reader does not have read_chip method")
                        
                        # Create synthetic PVP with timing information
                        num_vectors = data.shape[0] if isinstance(data, np.ndarray) and len(data.shape) > 0 else 512
                        pvp = {'TxTime': np.linspace(0, 1, num_vectors)}
                        
                        # Store data but don't process subapertures yet
                        estimator.raw_data = data
                        estimator.pvp = pvp
                        estimator.raw_data_image = np.abs(data)
                        estimator.data_loaded = True
                        
                        results['message'] = f'File {filename} loaded successfully'
                        # Create a visualization of the raw data
                        if hasattr(estimator, 'raw_data') and estimator.raw_data is not None:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            if hasattr(estimator, 'raw_data_image'):
                                ax.imshow(np.abs(estimator.raw_data_image), cmap='viridis', 
                                         aspect='auto', vmax=np.percentile(np.abs(estimator.raw_data_image), 95))
                                ax.set_title('Raw SAR Data (Amplitude)')
                                ax.set_xlabel('Range')
                                ax.set_ylabel('Azimuth')
                                plt.colorbar(ax.imshow(np.abs(estimator.raw_data_image), cmap='viridis', 
                                                     aspect='auto', vmax=np.percentile(np.abs(estimator.raw_data_image), 95)),
                                             ax=ax, label='Amplitude')
                                results['raw_data_plot'] = fig_to_base64(fig)
                                plt.close(fig)
                    except Exception as e:
                        error = f'Error loading file {filename}: {str(e)}'
    
    # Run subaperture creation step
    elif step == 'create_subapertures':
        if not hasattr(estimator, 'data_loaded') or not estimator.data_loaded:
            error = 'Data must be loaded first'
        else:
            try:
                # Use the dedicated create_subapertures method
                if estimator.create_subapertures():
                    results['message'] = f'Created {estimator.num_subapertures} subapertures successfully'
                    # Visualize the subapertures in frequency domain
                    if hasattr(estimator, 'subapertures') and estimator.subapertures:
                        fig, axs = plt.subplots(1, min(3, estimator.num_subapertures), figsize=(15, 5))
                        if estimator.num_subapertures == 1:
                            axs = [axs]
                        for i in range(min(3, estimator.num_subapertures)):
                            if hasattr(estimator.subapertures[i], 'shape'):
                                spec = np.fft.fftshift(np.fft.fft2(estimator.subapertures[i]))
                                axs[i].imshow(np.log10(np.abs(spec) + 1), cmap='inferno', aspect='auto')
                                axs[i].set_title(f'Subaperture {i+1} Spectrum')
                        plt.tight_layout()
                        results['subapertures_plot'] = fig_to_base64(fig)
                        plt.close(fig)
                else:
                    error = 'Error creating subapertures'
            except Exception as e:
                error = f'Error in create_subapertures: {str(e)}'
    
    # Run focus subapertures step
    elif step == 'focus_subapertures':
        if not hasattr(estimator, 'subapertures') or not estimator.subapertures:
            error = 'Subapertures must be created first'
        else:
            try:
                # Use the dedicated focus_subapertures method
                if estimator.focus_subapertures():
                    results['message'] = 'Subapertures focused successfully'
                    # Visualize the SLC images
                    if hasattr(estimator, 'slc_images') and estimator.slc_images:
                        fig, axs = plt.subplots(1, min(3, len(estimator.slc_images)), figsize=(15, 5))
                        if len(estimator.slc_images) == 1:
                            axs = [axs]
                        for i in range(min(3, len(estimator.slc_images))):
                            axs[i].imshow(np.abs(estimator.slc_images[i]), cmap='gray', aspect='auto',
                                        vmax=np.percentile(np.abs(estimator.slc_images[i]), 95))
                            axs[i].set_title(f'SLC Image {i+1}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(result_dir, 'slc_images.png'))
                        plt.close(fig)
                        update_processing_status('processing', 45, 'Subapertures focused successfully',
                                              console_log=f"Generated {len(estimator.slc_images)} SLC images")
                    
                    # TEMP: Skip displacement estimation for debug
                    logger.debug("TEMP: Skipping displacement estimation for debugging")
                    update_processing_status('complete', 100, 
                                          f'DEBUG MODE: Processing stopped after SLC image generation. Results saved to {os.path.basename(result_dir)}',
                                          console_log="DEBUG MODE: Stopped after SLC image generation")
                    return redirect(url_for('index'))
                else:
                    error = 'Error focusing subapertures'
            except Exception as e:
                error = f'Error in focus_subapertures: {str(e)}'
    
    # Run displacement estimation step
    elif step == 'estimate_displacement':
        if not hasattr(estimator, 'slc_images') or not estimator.slc_images:
            error = 'SLC images must be created first'
        else:
            try:
                # Monitor memory usage
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
                logger.info(f"Memory usage before displacement estimation: {memory_before:.2f} MB")
                
                # Use memory efficient version of displacement estimation
                update_processing_status('processing', 50, 'Estimating displacement (memory efficient)...',
                                     console_log="Estimating displacement using memory efficient algorithm")
                try:
                    logger.debug("Calling memory-efficient displacement estimation")
                    result = estimator.estimate_displacement_memory_efficient()
                    logger.debug(f"Displacement estimation returned: {result}")
                    
                    # Monitor memory after displacement estimation
                    memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
                    logger.info(f"Memory usage after displacement estimation: {memory_after:.2f} MB")
                    logger.info(f"Memory difference: {memory_after - memory_before:.2f} MB")
                    
                    if not result:
                        logger.error("Error in memory-efficient displacement estimation")
                        flash('Error estimating displacement', 'error') 
                        update_processing_status('error', 100, 'Failed to estimate displacement')
                        return redirect(url_for('index'))
                except Exception as e:
                    logger.error(f"Exception during displacement estimation: {str(e)}")
                    import traceback
                    trace = traceback.format_exc()
                    logger.error(f"Traceback: {trace}")
                    with open(os.path.join(result_dir, 'error_displacement.txt'), 'w') as f:
                        f.write(f"Error: {str(e)}\n\n{trace}")
                    flash('Error estimating displacement', 'error') 
                    update_processing_status('error', 100, f'Failed to estimate displacement: {str(e)}')
                    return redirect(url_for('index'))
                
                # Save displacement maps
                if hasattr(estimator, 'displacement_maps') and estimator.displacement_maps:
                    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Plot range displacement
                    if len(estimator.displacement_maps) > 0:
                        range_disp, azimuth_disp = estimator.displacement_maps[0]
                        im1 = axs[0, 0].imshow(range_disp, cmap='coolwarm', 
                                            vmin=-1, vmax=1, aspect='auto')
                        axs[0, 0].set_title('Range Displacement Map (frame 0)')
                        plt.colorbar(im1, ax=axs[0, 0], label='Displacement (pixels)')
                    
                        # Plot azimuth displacement
                        im2 = axs[0, 1].imshow(azimuth_disp, cmap='coolwarm', 
                                            vmin=-1, vmax=1, aspect='auto')
                        axs[0, 1].set_title('Azimuth Displacement Map (frame 0)')
                        plt.colorbar(im2, ax=axs[0, 1], label='Displacement (pixels)')
                    
                    # Plot empty placeholders for SNR and coherence maps
                    axs[1, 0].set_title('SNR Map (not implemented in memory-efficient mode)')
                    axs[1, 1].set_title('Coherence Map (not implemented in memory-efficient mode)')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_dir, 'displacement_maps.png'))
                    plt.close(fig)
                    update_processing_status('processing', 60, 'Displacement estimated successfully',
                                          console_log="Displacement maps calculated successfully")
            except Exception as e:
                error = f'Error in estimate_displacement: {str(e)}'
    
    # Run time series analysis step
    elif step == 'analyze_time_series':
        if not hasattr(estimator, 'displacement_maps') or not estimator.displacement_maps:
            error = 'Displacement maps must be created first'
        else:
            try:
                # Define measurement points
                if use_demo:
                    measurement_points = [(150, 200), (300, 350)]
                else:
                    # For real data, we could let user select points or use auto-detected points
                    measurement_points = [(100, 100), (200, 200)]
                
                session_data['measurement_points'] = measurement_points
                
                if estimator.analyze_time_series(measurement_points):
                    results['message'] = 'Time series analyzed successfully'
                    results['measurement_points'] = measurement_points
                    
                    # Visualize the time series
                    if hasattr(estimator, 'time_series') and estimator.time_series:
                        # For each measurement point, plot the time series
                        for i in range(len(measurement_points)):
                            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                            
                            # Plot range time series
                            axs[0, 0].plot(estimator.time_series['range'][i])
                            axs[0, 0].set_title(f'Range Displacement Time Series (Point {i})')
                            axs[0, 0].set_xlabel('Time (samples)')
                            axs[0, 0].set_ylabel('Displacement (pixels)')
                            axs[0, 0].grid(True)
                            
                            # Plot azimuth time series
                            axs[0, 1].plot(estimator.time_series['azimuth'][i])
                            axs[0, 1].set_title(f'Azimuth Displacement Time Series (Point {i})')
                            axs[0, 1].set_xlabel('Time (samples)')
                            axs[0, 1].set_ylabel('Displacement (pixels)')
                            axs[0, 1].grid(True)
                            
                            # Plot range frequency spectrum
                            freq, spectrum = estimator.frequency_spectra['range'][i]
                            axs[1, 0].plot(freq, spectrum)
                            axs[1, 0].set_title(f'Range Frequency Spectrum (Point {i})')
                            axs[1, 0].set_xlabel('Frequency (Hz)')
                            axs[1, 0].set_ylabel('Amplitude')
                            axs[1, 0].grid(True)
                            
                            # Plot azimuth frequency spectrum
                            freq, spectrum = estimator.frequency_spectra['azimuth'][i]
                            axs[1, 1].plot(freq, spectrum)
                            axs[1, 1].set_title(f'Azimuth Frequency Spectrum (Point {i})')
                            axs[1, 1].set_xlabel('Frequency (Hz)')
                            axs[1, 1].set_ylabel('Amplitude')
                            axs[1, 1].grid(True)
                            
                            plt.tight_layout()
                            results[f'time_series_plot_{i}'] = fig_to_base64(fig)
                            plt.close(fig)
                else:
                    error = 'Error analyzing time series'
            except Exception as e:
                error = f'Error in analyze_time_series: {str(e)}'
    
    # Run vibration energy calculation step
    elif step == 'calculate_vibration_energy':
        if not hasattr(estimator, 'time_series') or not estimator.time_series:
            error = 'Time series must be analyzed first'
        else:
            try:
                if estimator.calculate_vibration_energy_map():
                    results['message'] = 'Vibration energy map calculated successfully'
                    
                    # Visualize the vibration energy map
                    if hasattr(estimator, 'vibration_energy_map_db') and estimator.vibration_energy_map_db is not None:
                        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Plot the first SLC image for reference
                        if estimator.slc_images is not None and len(estimator.slc_images) > 0:
                            axs[0].imshow(np.abs(estimator.slc_images[0]), cmap='gray', aspect='auto')
                            axs[0].set_title('SLC Image (First Frame)')
                            axs[0].set_xlabel('Range (pixels)')
                            axs[0].set_ylabel('Azimuth (pixels)')
                        
                        # Plot vibration energy map
                        cmap = plt.cm.jet
                        norm = plt.Normalize(vmin=-25, vmax=0)
                        im = axs[1].imshow(estimator.vibration_energy_map_db, cmap=cmap, norm=norm, aspect='auto')
                        axs[1].set_title('Vibration Energy Map (dB)')
                        axs[1].set_xlabel('Range (pixels)')
                        axs[1].set_ylabel('Azimuth (pixels)')
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=axs[1])
                        cbar.set_label('Vibration Energy (dB)')
                        
                        plt.tight_layout()
                        results['vibration_energy_plot'] = fig_to_base64(fig)
                        plt.close(fig)
                        
                        # Also create a 3D visualization of the energy map
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # Create mesh grid for 3D plot
                        x = np.arange(0, estimator.vibration_energy_map_db.shape[1])
                        y = np.arange(0, estimator.vibration_energy_map_db.shape[0])
                        x, y = np.meshgrid(x, y)
                        
                        # Plot the surface
                        surf = ax.plot_surface(x, y, estimator.vibration_energy_map_db, 
                                              cmap='jet', linewidth=0, antialiased=False)
                        
                        ax.set_title('3D Vibration Energy Map')
                        ax.set_xlabel('Range (pixels)')
                        ax.set_ylabel('Azimuth (pixels)')
                        ax.set_zlabel('Energy (dB)')
                        
                        # Add colorbar
                        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                        
                        results['vibration_energy_3d_plot'] = fig_to_base64(fig)
                        plt.close(fig)
                else:
                    error = 'Error calculating vibration energy map'
            except Exception as e:
                error = f'Error in calculate_vibration_energy_map: {str(e)}'
    
    # Run ship region detection step
    elif step == 'detect_ship_regions':
        if not hasattr(estimator, 'vibration_energy_map_db') or estimator.vibration_energy_map_db is None:
            error = 'Vibration energy map must be calculated first'
        else:
            try:
                if estimator.detect_ship_regions(num_regions=3, energy_threshold=-15):
                    results['message'] = 'Ship regions detected successfully'
                    
                    # Use the ShipRegionDetector for more advanced detection
                    detector = ShipRegionDetector(min_region_size=20, energy_threshold=-15, num_regions=3)
                    ship_regions = detector.detect_regions(estimator.vibration_energy_map_db)
                    
                    # Get region statistics
                    region_stats = []
                    if ship_regions:
                        region_stats = detector.get_region_statistics(
                            estimator.vibration_energy_map_db, estimator.displacement_maps
                        )
                    
                    results['ship_regions'] = ship_regions
                    results['region_stats'] = region_stats
                    
                    # Visualize the ship regions
                    if ship_regions:
                        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Plot the first SLC image with regions
                        if estimator.slc_images is not None and len(estimator.slc_images) > 0:
                            axs[0].imshow(np.abs(estimator.slc_images[0]), cmap='gray', aspect='auto')
                            axs[0].set_title('SLC Image with Ship Regions')
                            axs[0].set_xlabel('Range (pixels)')
                            axs[0].set_ylabel('Azimuth (pixels)')
                        
                        # Plot vibration energy map with regions
                        cmap = plt.cm.jet
                        norm = plt.Normalize(vmin=-25, vmax=0)
                        im = axs[1].imshow(estimator.vibration_energy_map_db, cmap=cmap, norm=norm, aspect='auto')
                        axs[1].set_title('Vibration Energy Map with Ship Regions')
                        axs[1].set_xlabel('Range (pixels)')
                        axs[1].set_ylabel('Azimuth (pixels)')
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=axs[1])
                        cbar.set_label('Vibration Energy (dB)')
                        
                        # Add ship region labels and annotations
                        for region in ship_regions:
                            region_id = region['id']
                            centroid = region['centroid']
                            
                            # Find corresponding statistics
                            stats = next((s for s in region_stats if s['id'] == region_id), None)
                            
                            # Add label to SLC image
                            axs[0].text(centroid[1], centroid[0], str(region_id), 
                                      color='white', fontsize=12, ha='center', va='center',
                                      bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                            
                            # Add label to vibration energy map
                            axs[1].text(centroid[1], centroid[0], str(region_id), 
                                      color='white', fontsize=12, ha='center', va='center',
                                      bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                            
                            # Add arrows pointing to the regions
                            arrow_length = 40
                            arrow_angle = np.random.uniform(0, 2*np.pi)
                            arrow_dx = arrow_length * np.cos(arrow_angle)
                            arrow_dy = arrow_length * np.sin(arrow_angle)
                            arrow_start_x = centroid[1] + arrow_dx
                            arrow_start_y = centroid[0] + arrow_dy
                            
                            # For SLC image
                            axs[0].annotate('', xy=(centroid[1], centroid[0]), 
                                          xytext=(arrow_start_x, arrow_start_y),
                                          arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
                            
                            # For vibration energy map
                            axs[1].annotate('', xy=(centroid[1], centroid[0]), 
                                          xytext=(arrow_start_x, arrow_start_y),
                                          arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
                            
                            # Add statistics annotation
                            if stats:
                                stats_text = f"Region {region_id}:\n"
                                stats_text += f"Energy: {stats['mean_energy']:.1f} dB\n"
                                if 'mean_range_disp' in stats:
                                    stats_text += f"Range Disp: {stats['mean_range_disp']:.3f} px\n"
                                    stats_text += f"Azimuth Disp: {stats['mean_azimuth_disp']:.3f} px"
                                
                                # Position the text box near the arrow start
                                text_x = arrow_start_x + 10 * np.cos(arrow_angle)
                                text_y = arrow_start_y + 10 * np.sin(arrow_angle)
                                
                                # Add text to both plots
                                axs[0].annotate(stats_text, xy=(text_x, text_y), 
                                              color='white', fontsize=8,
                                              bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                                
                                axs[1].annotate(stats_text, xy=(text_x, text_y), 
                                              color='white', fontsize=8,
                                              bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                        
                        plt.tight_layout()
                        results['ship_regions_plot'] = fig_to_base64(fig)
                        plt.close(fig)
                else:
                    error = 'Error detecting ship regions'
            except Exception as e:
                error = f'Error in detect_ship_regions: {str(e)}'
    else:
        error = f'Unknown step: {step}'
    
    # Add error if there is one
    if error:
        results['error'] = error
    
    return jsonify({
        'success': error is None,
        'error': error,
        'results': results,
        'session': session_data
    })

@app.route('/process', methods=['POST'])
def process():
    """
    Process uploaded CPHD file or demo data and analyze micro-motion.

    Returns:
    - Flask response, redirects to index with results or error
    """
    # Create results directory with timestamp
    result_dir = create_results_folder()
    
    # Setup logging for this processing run
    process_log_file = os.path.join(result_dir, 'processing.log')
    file_handler = logging.FileHandler(process_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create a "detailed_logs.txt" file that will contain all console output
    detailed_log_path = os.path.join(result_dir, 'detailed_logs.txt')
    with open(detailed_log_path, 'w') as f:
        f.write(f"=== Processing started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    logger.info(f"Starting new processing run, results will be saved to {result_dir}")
    
    # Initialize processing status
    update_processing_status('starting', 0, 'Initializing processing', 
                           console_log=f"Starting new processing run, results directory: {result_dir}")
    
    try:
        # Placeholder for estimator initialization
        from micromotion_estimator import ShipMicroMotionEstimator
        num_subapertures = int(request.form.get('num_subapertures', 7))
        window_size = int(request.form.get('window_size', 64))
        overlap = float(request.form.get('overlap', 0.5))
        
        init_msg = f"Initializing estimator with parameters: num_subapertures={num_subapertures}, window_size={window_size}, overlap={overlap}"
        logger.info(init_msg)
        
        # Function to append to the detailed logs file
        def append_to_detailed_log(message):
            with open(detailed_log_path, 'a') as f:
                f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")
            
            # Immediately update session with this log message
            current_status = session.get('processing_status', {})
            current_logs = current_status.get('log_messages', [])
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"{timestamp}: {message}"
            current_logs.append(log_entry)
            current_status['log_messages'] = current_logs[-10:] # Keep last 10
            current_status['details'] = message  # Also update the current status details
            session['processing_status'] = current_status
            session.modified = True
            
            return message
            
        append_to_detailed_log(init_msg)
        
        estimator = ShipMicroMotionEstimator(num_subapertures=num_subapertures, 
                                             window_size=window_size, 
                                                 overlap=overlap,
                                                 debug_mode=True,  # Enable debug mode
                                                 log_callback=append_to_detailed_log)  # Pass callback for logging

        use_demo = request.form.get('use_demo', 'false') == 'true'
        
        # Calculate total processing time
        total_time = sum(STEP_DURATIONS.values())
        
        if use_demo:
            update_processing_status('loading', 5, 'Generating synthetic demo data',
                                  console_log="Using synthetic demo data")
            
            if estimator.load_data('demo'):
                update_processing_status('loading', 10, 'Synthetic data generated successfully',
                                     console_log="Successfully created synthetic data")
                measurement_points = [(150, 200), (300, 350)]
            else:
                logger.error("Error creating synthetic data")
                flash('Error creating synthetic data', 'error')
                update_processing_status('error', 100, 'Failed to create synthetic data')
                return redirect(url_for('index'))
        else:
            # Check for existing file selection first
            existing_file = request.form.get('existing_file', '')
            
            if existing_file:
                # Use existing file from the uploads directory
                cphd_file_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_file)
                if not os.path.exists(cphd_file_path):
                    logger.error(f"Selected file {existing_file} not found")
                    flash(f'Selected file {existing_file} not found', 'error')
                    update_processing_status('error', 100, f'File not found: {existing_file}')
                    return redirect(url_for('index'))
                
                update_processing_status('loading', 5, f'Using existing file: {existing_file}',
                                     console_log=f"Using existing file: {existing_file}")
                flash(f'Using existing file: {existing_file}', 'info')
            else:
                # Handle new file upload
                file = request.files.get('file')
                if not file or file.filename == '':
                    logger.error("No file uploaded")
                    flash('No file uploaded', 'error')
                    update_processing_status('error', 100, 'No file uploaded')
                    return redirect(url_for('index'))
                
                cphd_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                update_processing_status('loading', 5, f'Saving uploaded file: {file.filename}',
                                      console_log=f"Saving uploaded file to {cphd_file_path}")
                file.save(cphd_file_path)
                flash(f'File {file.filename} uploaded successfully', 'info')
            
            # Process the file (either existing or newly uploaded)
            update_processing_status('loading', 10, f'Loading CPHD file: {os.path.basename(cphd_file_path)}',
                                  console_log=f"Loading CPHD file: {cphd_file_path}")
            
            # This could take time, so let's update the status to show it's still working
            def progress_callback(step, message):
                progress = 10 + ((step / 4) * 15)  # Scale to 10-25% range during loading
                update_processing_status('loading', int(progress), message)
                append_to_detailed_log(message)
                
            # Set incremental progress updates during load
            load_steps = ["Opening CPHD file", "Reading metadata", "Reading data chip", "Processing PVP data"]
            for i, step in enumerate(load_steps):
                progress_callback(i, f"CPHD Loading: {step}...")
                time.sleep(0.5)  # Small delay to ensure UI updates
            
            if estimator.load_data(cphd_file_path):
                update_processing_status('processing', 25, 'CPHD file loaded successfully', 
                                      console_log="CPHD file loaded successfully")
                
                # Save raw data visualization to results folder
                if hasattr(estimator, 'raw_data_image') and estimator.raw_data_image is not None:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(np.abs(estimator.raw_data_image), cmap='viridis', 
                             aspect='auto', vmax=np.percentile(np.abs(estimator.raw_data_image), 95))
                    ax.set_title('Raw SAR Data (Amplitude)')
                    ax.set_xlabel('Range')
                    ax.set_ylabel('Azimuth')
                    plt.colorbar(ax.imshow(np.abs(estimator.raw_data_image), cmap='viridis', 
                                         aspect='auto', vmax=np.percentile(np.abs(estimator.raw_data_image), 95)),
                                 ax=ax, label='Amplitude')
                    plt.savefig(os.path.join(result_dir, 'raw_data.png'))
                    plt.close(fig)
                    update_processing_status('processing', 25, 'Raw data visualization saved',
                                          console_log="Saved raw data visualization to results folder")
                
                update_processing_status('processing', 30, 'Creating subapertures...',
                                      console_log="Creating subapertures")
                if not estimator.create_subapertures():
                    logger.error("Error creating subapertures")
                    flash('Error creating subapertures', 'error')
                    update_processing_status('error', 100, 'Failed to create subapertures')
                    
                    # Save what we have so far to the results directory
                    with open(os.path.join(result_dir, 'error_details.txt'), 'w') as f:
                        f.write("Error occurred during subaperture creation\n")
                        f.write(f"Raw data shape: {estimator.raw_data.shape if hasattr(estimator, 'raw_data') and hasattr(estimator.raw_data, 'shape') else 'unknown'}\n")
                        f.write(f"Subapertures: {len(estimator.subapertures) if hasattr(estimator, 'subapertures') and estimator.subapertures else 'not created'}\n")
                        
                        # Add traceback if available
                        if hasattr(estimator, 'last_error') and estimator.last_error:
                            f.write(f"\nError details:\n{estimator.last_error}\n")
                    
                    # Log the state of each processing step to help with debugging
                    logger.debug(f"Data loaded: {estimator.data_loaded if hasattr(estimator, 'data_loaded') else 'unknown'}")
                    logger.debug(f"Raw data shape: {estimator.raw_data.shape if hasattr(estimator, 'raw_data') and hasattr(estimator.raw_data, 'shape') else 'unknown'}")
                    logger.debug(f"Subapertures: {len(estimator.subapertures) if hasattr(estimator, 'subapertures') and estimator.subapertures else 'not created'}")
                    
                    return redirect(url_for('index'))
                
                # Save subaperture visualization
                if hasattr(estimator, 'subapertures') and estimator.subapertures:
                    fig, axs = plt.subplots(1, min(3, len(estimator.subapertures)), figsize=(15, 5))
                    if len(estimator.subapertures) == 1:
                        axs = [axs]
                    for i in range(min(3, len(estimator.subapertures))):
                        spec = np.fft.fftshift(np.fft.fft2(estimator.subapertures[i]))
                        axs[i].imshow(np.log10(np.abs(spec) + 1), cmap='inferno', aspect='auto')
                        axs[i].set_title(f'Subaperture {i+1} Spectrum')
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_dir, 'subapertures_spectrum.png'))
                    plt.close(fig)
                    update_processing_status('processing', 35, 'Subapertures created successfully', 
                                          console_log=f"Created {len(estimator.subapertures)} subapertures")
                
                update_processing_status('processing', 40, 'Focusing subapertures...',
                                      console_log="Focusing subapertures")
                if not estimator.focus_subapertures():
                    logger.error("Error focusing subapertures")
                    flash('Error focusing subapertures', 'error')
                    update_processing_status('error', 100, 'Failed to focus subapertures')
                    return redirect(url_for('index'))
                
                # Save SLC images
                if hasattr(estimator, 'slc_images') and estimator.slc_images:
                    fig, axs = plt.subplots(1, min(3, len(estimator.slc_images)), figsize=(15, 5))
                    if len(estimator.slc_images) == 1:
                        axs = [axs]
                    for i in range(min(3, len(estimator.slc_images))):
                        axs[i].imshow(np.abs(estimator.slc_images[i]), cmap='gray', aspect='auto',
                                    vmax=np.percentile(np.abs(estimator.slc_images[i]), 95))
                        axs[i].set_title(f'SLC Image {i+1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_dir, 'slc_images.png'))
                    plt.close(fig)
                    update_processing_status('processing', 45, 'Subapertures focused successfully',
                                          console_log=f"Generated {len(estimator.slc_images)} SLC images")
                
                # Monitor memory usage
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
                logger.info(f"Memory usage before displacement estimation: {memory_before:.2f} MB")
                
                # Use memory efficient version of displacement estimation
                update_processing_status('processing', 50, 'Estimating displacement (memory efficient)...',
                                     console_log="Estimating displacement using memory efficient algorithm")
                try:
                    logger.debug("Calling memory-efficient displacement estimation")
                    result = estimator.estimate_displacement_memory_efficient()
                    logger.debug(f"Displacement estimation returned: {result}")
                    
                    # Monitor memory after displacement estimation
                    memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
                    logger.info(f"Memory usage after displacement estimation: {memory_after:.2f} MB")
                    logger.info(f"Memory difference: {memory_after - memory_before:.2f} MB")
                    
                    if not result:
                        logger.error("Error in memory-efficient displacement estimation")
                        flash('Error estimating displacement', 'error') 
                        update_processing_status('error', 100, 'Failed to estimate displacement')
                        return redirect(url_for('index'))
                except Exception as e:
                    logger.error(f"Exception during displacement estimation: {str(e)}")
                    import traceback
                    trace = traceback.format_exc()
                    logger.error(f"Traceback: {trace}")
                    with open(os.path.join(result_dir, 'error_displacement.txt'), 'w') as f:
                        f.write(f"Error: {str(e)}\n\n{trace}")
                    flash('Error estimating displacement', 'error') 
                    update_processing_status('error', 100, f'Failed to estimate displacement: {str(e)}')
                    return redirect(url_for('index'))
                
                # Save displacement maps
                if hasattr(estimator, 'displacement_maps') and estimator.displacement_maps:
                    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Plot range displacement
                    if len(estimator.displacement_maps) > 0:
                        range_disp, azimuth_disp = estimator.displacement_maps[0]
                        im1 = axs[0, 0].imshow(range_disp, cmap='coolwarm', 
                                            vmin=-1, vmax=1, aspect='auto')
                        axs[0, 0].set_title('Range Displacement Map (frame 0)')
                        plt.colorbar(im1, ax=axs[0, 0], label='Displacement (pixels)')
                    
                        # Plot azimuth displacement
                        im2 = axs[0, 1].imshow(azimuth_disp, cmap='coolwarm', 
                                            vmin=-1, vmax=1, aspect='auto')
                        axs[0, 1].set_title('Azimuth Displacement Map (frame 0)')
                        plt.colorbar(im2, ax=axs[0, 1], label='Displacement (pixels)')
                    
                    # Plot empty placeholders for SNR and coherence maps
                    axs[1, 0].set_title('SNR Map (not implemented in memory-efficient mode)')
                    axs[1, 1].set_title('Coherence Map (not implemented in memory-efficient mode)')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_dir, 'displacement_maps.png'))
                    plt.close(fig)
                    update_processing_status('processing', 60, 'Displacement estimated successfully',
                                          console_log="Displacement maps calculated successfully")
            else:
                logger.error("Error loading CPHD file")
                flash('Error loading CPHD file', 'error')
                update_processing_status('error', 100, 'Failed to load CPHD file')
                return redirect(url_for('index'))
        
        # Analyze time series
        update_processing_status('processing', 90, 'Analyzing time series at measurement points...',
                              console_log=f"Analyzing time series at {len(measurement_points)} measurement points")
        
        if not estimator.analyze_time_series(measurement_points):
            logger.error("Error analyzing time series")
            flash('Error analyzing time series', 'error')
            update_processing_status('error', 100, 'Failed to analyze time series')
            return redirect(url_for('index'))
        
        # Save all results to the timestamped directory
        update_processing_status('saving', 95, 'Saving results to timestamped folder...',
                              console_log=f"Saving results to {result_dir}")
        
        # Save vibration energy map
        try:
            energy_map_file = os.path.join(result_dir, 'vibration_energy_map.png')
            estimator.plot_vibration_energy_map(output_file=energy_map_file)
            append_to_detailed_log(f"Saved vibration energy map to {energy_map_file}")
        except Exception as e:
            logger.error(f"Error saving vibration energy map: {str(e)}")
            append_to_detailed_log(f"Error saving vibration energy map: {str(e)}")
        
        # Save individual measurement point results
        for i in range(len(measurement_points)):
            try:
                point_result_file = os.path.join(result_dir, f'point_{i}_results.png')
                estimator.plot_results(i, output_file=point_result_file)
                append_to_detailed_log(f"Saved point {i} results to {point_result_file}")
            except Exception as e:
                logger.error(f"Error saving point {i} results: {str(e)}")
                append_to_detailed_log(f"Error saving point {i} results: {str(e)}")
        
        # Save metadata about the processing
        metadata = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                'num_subapertures': num_subapertures,
                'window_size': window_size,
                'overlap': overlap
            },
            'file': os.path.basename(cphd_file_path) if not use_demo else 'demo',
            'measurement_points': [list(mp) for mp in measurement_points],
            'processing_time': datetime.datetime.now().strftime("%H:%M:%S")
        }
        
        with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a human-readable summary
        with open(os.path.join(result_dir, 'summary.txt'), 'w') as f:
            f.write(f"Processing completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {os.path.basename(cphd_file_path) if not use_demo else 'demo'}\n")
            f.write(f"Parameters: num_subapertures={num_subapertures}, window_size={window_size}, overlap={overlap}\n")
            f.write(f"Measurement points:\n")
            for i, point in enumerate(measurement_points):
                f.write(f"  Point {i}: {point}\n")
            f.write("\nResults files:\n")
            for filename in os.listdir(result_dir):
                file_path = os.path.join(result_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    f.write(f"  {filename} ({file_size/1024:.1f} KB)\n")
        
        # Update completion status with results directory
        update_processing_status('complete', 100, 
                              f'Processing completed. Results saved to {os.path.basename(result_dir)}',
                              console_log="Processing completed successfully")
        flash(f'Processing completed successfully. Results saved to {os.path.basename(result_dir)}', 'success')
        
    except Exception as e:
        logger.exception(f"Unexpected error during processing: {str(e)}")
        
        # Write the exception to the detailed log
        with open(detailed_log_path, 'a') as f:
            f.write(f"\n\nERROR: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        
        flash(f'Error during processing: {str(e)}', 'error')
        update_processing_status('error', 100, f'Error: {str(e)}')
    
    finally:
        # Remove the file handler
        logger.removeHandler(file_handler)
        
        # Write final status to the detailed log
        with open(detailed_log_path, 'a') as f:
            f.write(f"\n=== Processing finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    return redirect(url_for('index'))

@app.route('/results/<path:filename>')
def results_file(filename):
    """Serve files from the results directory"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
