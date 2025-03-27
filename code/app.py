from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import datetime
import logging
import json
import fnmatch
from werkzeug.utils import secure_filename
from micromotion_estimator import ShipMicroMotionEstimator
from ship_region_detector import ShipRegionDetector
from test_estimator import create_synthetic_data
from flask_session import Session  # Import Flask-Session
from flask_socketio import SocketIO, emit  # Import Flask-SocketIO
import time
import psutil
import re
import threading

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

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   ping_timeout=60, 
                   ping_interval=25,
                   async_mode='threading',  # Use threading mode for better performance with Flask
                   logger=False,            # Disable default SocketIO logging
                   engineio_logger=False)   # Disable engineIO logging

# Create upload and results folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  # Create session directory

# Allowed file extensions
ALLOWED_CPHD_PATTERNS = ['*.cphd', '*.CPHD']
ALLOWED_SICD_PATTERNS = ['*.nitf', '*.NTF', '*.NITF', '*.ntf']
ALLOWED_PATTERNS = ALLOWED_CPHD_PATTERNS + ALLOWED_SICD_PATTERNS

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    for pattern in ALLOWED_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

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
    """Update the processing status in the session and emit via WebSocket"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # If console_log is provided, log it to console and file
    if console_log:
        logger.info(console_log)
    
    # Get current status to preserve logs
    current_status = session.get('processing_status', {})
    log_messages = current_status.get('log_messages', [])
    
    # Preserve progress if not explicitly provided but exists in current status
    if progress is None and 'progress' in current_status:
        progress = current_status.get('progress')
    
    # Create new status dict with updated values, preserving existing values if not provided
    updated_status = {
        'status': status,
        'progress': progress,
        'details': details,
        'timestamp': timestamp,
        'log_messages': log_messages
    }
    
    # Add the new message to the log history (keep last 15 messages)
    if details:
        log_entry = f"{timestamp}: {details}"
        updated_status['log_messages'].append(log_entry)
        # Keep only the most recent 15 messages
        updated_status['log_messages'] = updated_status['log_messages'][-15:]
    
    # Ensure status transitions make sense
    if status == 'complete' and progress is not None and progress < 100:
        # If marking as complete, ensure progress is 100%
        updated_status['progress'] = 100
        logger.debug("Auto-adjusted progress to 100% for 'complete' status")
    
    # Update the session
    session['processing_status'] = updated_status
    session.modified = True
    
    # Immediately commit the session - this is important to ensure updates are visible
    if hasattr(session, 'save_session'):
        session.save_session(None)  # Force session save
    
    # Emit WebSocket event with the updated status - use socketio.emit for broadcast to all clients
    try:
        # Broadcast to all connected clients
        socketio.emit('status_update', updated_status, namespace='/')
        logger.debug(f"WebSocket: Broadcasted status_update event: {status}, progress: {progress}")
    except Exception as e:
        logger.error(f"Error emitting WebSocket event: {e}")
    
    logger.debug(f"Updated processing status: {status}, progress: {progress}, details: {details}")

# Add a function to copy console output to the processing log
def log_to_processing(message):
    """Log a message to both console and processing log"""
    logger.info(message)
    return message

# Function to append to the detailed logs file - defined at module level
def append_to_detailed_log(message):
    """Append a message to the detailed log file and log to console.
    This function does NOT modify the session status - use update_processing_status for that."""
    try:
        # Get result directory from session if available
        result_dir = session.get('result_dir', None)
        if result_dir:
            # Construct the detailed log path
            detailed_log_path = os.path.join(app.config['RESULTS_FOLDER'], result_dir, 'detailed_logs.txt')
            # Write to the log file
            with open(detailed_log_path, 'a') as f:
                f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")
        else:
            logger.warning(f"Cannot write to detailed log: result_dir not in session for message: {message}")
        
        # Log to console in any case
        logger.info(message)
    except Exception as e:
        logger.error(f"Error writing to detailed log: {e}")
    
    return message

# New function that updates both detailed log and UI status
def estimator_log_callback(message):
    """
    Callback function for ShipMicroMotionEstimator log messages
    that updates both the detailed log and the processing status UI
    """
    try:
        # First append to detailed log
        append_to_detailed_log(message)
        
        # Get current status to preserve status/progress
        current_status = session.get('processing_status', {})
        current_status_type = current_status.get('status', 'processing')
        current_progress = current_status.get('progress', 50)
        
        # Extract progress indicators from the message if present
        # Look for patterns like "75% complete" or "Progress: 80%"
        progress_match = re.search(r'(\d+)%\s+(complete|done|finished)', message, re.IGNORECASE)
        if progress_match:
            extracted_progress = int(progress_match.group(1))
            # Only update if it's greater than current progress
            if extracted_progress > current_progress:
                current_progress = extracted_progress
                logger.debug(f"Updated progress to {current_progress}% from message")
        
        # Make sure we don't prematurely set status to complete or error
        # Only the main process function should set these final states
        if current_status_type not in ['idle', 'starting', 'loading', 'processing', 'saving']:
            # We're likely in 'complete' or 'error' state, but still getting log messages
            # Keep the status as is but ensure UI shows latest messages
            pass
        
        # Add message to log_messages without changing status/progress
        if 'log_messages' not in current_status:
            current_status['log_messages'] = []
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp}: {message}"
        current_status['log_messages'].append(log_entry)
        # Keep only most recent 15 messages
        current_status['log_messages'] = current_status['log_messages'][-15:]
        
        # Update the progress in the status dictionary
        current_status['progress'] = current_progress
        current_status['details'] = str(message)
        current_status['timestamp'] = timestamp
        
        # Save the updated status with new log message
        session['processing_status'] = current_status
        session.modified = True
        
        # Force session save to ensure updates are visible immediately
        if hasattr(session, 'save_session'):
            session.save_session(None)
        
        # Emit WebSocket event with the updated status - use socketio.emit for broadcast to all clients
        try:
            # Broadcast to all connected clients
            socketio.emit('status_update', current_status, namespace='/')
            logger.debug(f"WebSocket: Log callback broadcasted status update")
        except Exception as e:
            logger.error(f"Error emitting WebSocket event from log callback: {e}")
            
    except Exception as e:
        logger.error(f"Error in estimator log callback: {e}")
    
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
    # Always reset processing status to idle on application startup
    session['processing_status'] = {
        'status': 'idle',
        'progress': None,
        'details': 'Ready to process',
        'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
        'log_messages': []
    }
    session.modified = True
    
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
    status = session.get('processing_status', {'status': 'idle'})
    # Add more detailed debug logging
    current_status = status.get('status', 'unknown')
    current_progress = status.get('progress', 0)
    current_details = status.get('details', 'No details')
    log_count = len(status.get('log_messages', []))
    
    logger.debug(f"Status check: status={current_status}, progress={current_progress}, details='{current_details}', log_count={log_count}")
    return jsonify(status)

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
    Process uploaded CPHD/SICD file or demo data and analyze micro-motion.

    Returns:
    - Flask response, redirects to index with results or error
    """
    # Create a new results folder for this run
    result_dir = create_results_folder()
    
    # Flag to track processing success
    has_error = False
    
    # Store the result directory name in the session for append_to_detailed_log function
    result_dir_name = os.path.basename(result_dir)
    session['result_dir'] = result_dir_name
    
    # Reset the processing status to ensure we don't show stale data from previous runs
    session['processing_status'] = {
        'status': 'starting',
        'progress': 0,
        'details': 'Initializing processing...',
        'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
        'log_messages': []
    }
    session.modified = True
    # Force session save to ensure the reset takes effect immediately
    if hasattr(session, 'save_session'):
        session.save_session(None)
    
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
    
    # Log the initial message
    initial_msg = f"Starting new processing run, results directory: {result_dir}"
    logger.info(initial_msg)
    
    # Use the global append_to_detailed_log to log subsequent messages
    append_to_detailed_log("Process initialization complete")
    update_processing_status('starting', 0, 'Initializing processing',
                           console_log=f"Starting new processing run, results directory: {result_dir}")

    try:
        # Get parameters from form
        num_subapertures = int(request.form.get('num_subapertures', 7))
        window_size = int(request.form.get('window_size', 64))
        overlap = float(request.form.get('overlap', 0.5))
        min_region_size = int(request.form.get('min_region_size', 5))
        energy_threshold = float(request.form.get('energy_threshold', -15))
        num_regions_to_detect = int(request.form.get('num_regions_to_detect', 3))
        
        # Log received parameters
        append_to_detailed_log(f"Parameters: num_subapertures={num_subapertures}, window_size={window_size}, overlap={overlap}")
        append_to_detailed_log(f"Region detection parameters: min_size={min_region_size}, threshold={energy_threshold}, num_regions={num_regions_to_detect}")
        
        # Initialize data type, will be set during load_data
        data_type = "unknown"
        
        # Instantiate estimator with log callback that updates both detailed log and UI status
        estimator = ShipMicroMotionEstimator(num_subapertures=num_subapertures,
                                             window_size=window_size,
                                             overlap=overlap,
                                             debug_mode=True,
                                             log_callback=estimator_log_callback)
        
        # Initialize the ship region detector with the same log callback
        detector = ShipRegionDetector(min_region_size=min_region_size,
                                    energy_threshold=energy_threshold,
                                    num_regions=num_regions_to_detect,
                                    log_callback=estimator_log_callback)
        
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
                
                # Create subapertures and SLC images
                update_processing_status('processing', 30, 'Creating subapertures...',
                                      console_log="Creating subapertures for demo data")
                if not estimator.create_subapertures():
                    logger.error("Error creating subapertures for demo data")
                    flash('Error creating subapertures', 'error')
                    update_processing_status('error', 100, 'Failed to create subapertures for demo data')
                    return redirect(url_for('index'))
                
                update_processing_status('processing', 40, 'Focusing subapertures...',
                                      console_log="Focusing subapertures for demo data")
                if not estimator.focus_subapertures():
                    logger.error("Error focusing subapertures for demo data")
                    flash('Error focusing subapertures', 'error')
                    update_processing_status('error', 100, 'Failed to focus subapertures for demo data')
                    return redirect(url_for('index'))
                
                # Estimate displacement
                update_processing_status('processing', 50, 'Estimating displacement for demo data...',
                                     console_log="Estimating displacement for demo data")
                if not estimator.estimate_displacement_enhanced():
                    logger.error("Error estimating displacement for demo data")
                    flash('Error estimating displacement for demo data', 'error')
                    update_processing_status('error', 100, 'Failed to estimate displacement for demo data')
                    return redirect(url_for('index'))
                
                update_processing_status('processing', 60, 'Displacement estimated successfully for demo data',
                                      console_log="Displacement maps calculated successfully for demo data")
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
                filename = existing_file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if not os.path.exists(file_path):
                    logger.error(f"Selected file {filename} not found")
                    flash(f'Selected file {filename} not found', 'error')
                    update_processing_status('error', 100, f'File not found: {filename}')
                    return redirect(url_for('index'))
                
                update_processing_status('loading', 5, f'Using existing file: {filename}',
                                     console_log=f"Using existing file: {filename}")
                flash(f'Using existing file: {filename}', 'info')
            else:
                # Handle new file upload
                file = request.files.get('file')
                if not file or file.filename == '':
                    logger.error("No file uploaded")
                    flash('No file uploaded', 'error')
                    update_processing_status('error', 100, 'No file uploaded')
                    return redirect(url_for('index'))
                
                filename = secure_filename(file.filename)
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
                    file.save(file_path)
                    flash(f'File {filename} uploaded successfully', 'info')
                except Exception as e:
                    msg = f"Error saving uploaded file: {e}"
                    logger.error(msg)
                    flash(msg, 'error')
                    update_processing_status('error', 100, msg)
                    return redirect(url_for('index'))
            
            # Process the file (either existing or newly uploaded)
            update_processing_status('loading', 10, f'Loading data file: {os.path.basename(file_path)}',
                                  console_log=f"Loading data file: {file_path}")
            
            # This could take time, so let's update the status to show it's still working
            def progress_callback(step, message):
                progress = 10 + ((step / 4) * 15)  # Scale to 10-25% range during loading
                # Log to detailed log using global function
                append_to_detailed_log(message)
                # Update UI status with the same message
                update_processing_status('loading', int(progress), message)
                
            # Set incremental progress updates during load
            load_steps = ["Opening file", "Reading metadata", "Reading data chip", "Processing auxiliary data"]
            for i, step in enumerate(load_steps):
                progress_callback(i, f"Loading: {step}...")
                time.sleep(0.5)  # Small delay to ensure UI updates
            
            if estimator.load_data(file_path):
                data_type_display = estimator.data_type.upper() if hasattr(estimator, 'data_type') and estimator.data_type else "DATA"
                update_processing_status('processing', 25, f'{data_type_display} file loaded successfully', 
                                      console_log=f"{data_type_display} file loaded successfully")
                
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
                    logger.error("Error focusing subapertures for demo data")
                    flash('Error focusing subapertures', 'error')
                    update_processing_status('error', 100, 'Failed to focus subapertures for demo data')
                    has_error = True
                    
                # Only continue if no errors so far
                if not has_error:
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
                    
                    # Try displacement estimation
                    try:
                        logger.debug("Calling enhanced memory-efficient displacement estimation")
                        
                        # First update the status to ensure it's visible before starting the thread
                        update_processing_status('processing', 50, 'Starting displacement estimation...',
                                              console_log="Beginning enhanced memory-efficient displacement estimation")
                        
                        # Set up a progress monitor for long-running displacement estimation
                        def displacement_progress_monitor():
                            """Function to periodically update UI during long-running displacement estimation"""
                            # Define more precise progress points from 52% to 90%
                            progress_points = [52, 55, 60, 65, 70, 75, 80, 85, 90, 95]
                            for p in progress_points:
                                time.sleep(2)  # Wait a few seconds between updates
                                # Use a format that our progress parser can detect
                                estimator_log_callback(f"Displacement estimation {p}% complete...")
                        
                        # Start progress monitor in background thread
                        progress_thread = threading.Thread(target=displacement_progress_monitor)
                        progress_thread.daemon = True
                        progress_thread.start()

                        # Run the actual displacement estimation
                        # Let the estimator's own progress updates provide UI feedback
                        result = estimator.estimate_displacement_enhanced()
                        logger.debug(f"Displacement estimation returned: {result}")
                        
                        # Monitor memory after displacement estimation
                        memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
                        logger.info(f"Memory usage after displacement estimation: {memory_after:.2f} MB")
                        logger.info(f"Memory difference: {memory_after - memory_before:.2f} MB")
                        
                        if not result:
                            logger.error("Error in enhanced memory-efficient displacement estimation")
                            flash('Error estimating displacement', 'error') 
                            update_processing_status('error', 100, 'Failed to estimate displacement')
                            has_error = True
                    except Exception as e:
                        logger.error(f"Exception during displacement estimation: {str(e)}")
                        import traceback
                        trace = traceback.format_exc()
                        logger.error(f"Traceback: {trace}")
                        with open(os.path.join(result_dir, 'error_displacement.txt'), 'w') as f:
                            f.write(f"Error: {str(e)}\n\n{trace}")
                        flash('Error estimating displacement', 'error') 
                        update_processing_status('error', 100, f'Failed to estimate displacement: {str(e)}')
                        has_error = True
                    
                    # Only proceed with saving displacement maps if no errors occurred
                    if not has_error and hasattr(estimator, 'displacement_maps') and estimator.displacement_maps:
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
                        axs[1, 0].set_title('SNR Map (not implemented in enhanced mode)')
                        axs[1, 1].set_title('Coherence Map (not implemented in enhanced mode)')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(result_dir, 'displacement_maps.png'))
                        plt.close(fig)
                        update_processing_status('processing', 60, 'Displacement estimated successfully',
                                              console_log="Displacement maps calculated successfully")
                    
                    # Initialize measurement points for non-demo case
                    if 'measurement_points' not in locals():
                        # Choose default measurement points based on the size of displacement maps
                        if hasattr(estimator, 'displacement_maps') and estimator.displacement_maps and len(estimator.displacement_maps) > 0:
                            range_disp, _ = estimator.displacement_maps[0]
                            if range_disp is not None:
                                height, width = range_disp.shape
                                measurement_points = [(width//4, height//4), (width*3//4, height*3//4)]
                                mp_msg = f"Defined default measurement points based on image size {range_disp.shape}: {measurement_points}"
                                append_to_detailed_log(mp_msg)
                                update_processing_status('processing', 65, mp_msg)
                            else:
                                measurement_points = [(100, 100), (200, 200)]  # Default fallback
                                mp_msg = f"Could not determine displacement size, using fallback measurement points: {measurement_points}"
                                append_to_detailed_log(mp_msg)
                                update_processing_status('processing', 65, mp_msg)
                        else:
                            measurement_points = [(100, 100), (200, 200)]  # Default fallback
                            mp_msg = f"No displacement maps available, using fallback measurement points: {measurement_points}"
                            append_to_detailed_log(mp_msg)
                            update_processing_status('processing', 65, mp_msg)
            else:
                error_msg = f"Error loading data file: {filename}"
                logger.error(error_msg)
                if estimator.last_error:
                     error_msg += f" Details: {estimator.last_error.splitlines()[0]}" # Show first line of error
                flash(error_msg, 'error')
                update_processing_status('error', 100, f'Failed to load data file: {error_msg}')
                # Save error details if available
                if estimator.last_error:
                     with open(os.path.join(result_dir, 'error_loading.txt'), 'w') as f:
                          f.write(estimator.last_error)
                has_error = True
        
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
            # Calculate vibration energy map first
            update_processing_status('processing', 92, 'Calculating vibration energy map...',
                                  console_log="Calculating vibration energy map")
            
            if estimator.calculate_vibration_energy_map():
                energy_map_file = os.path.join(result_dir, 'vibration_energy_map.png')
                estimator.plot_vibration_energy_map(output_file=energy_map_file)
                save_msg = f"Saved vibration energy map to {energy_map_file}"
                append_to_detailed_log(save_msg)
                update_processing_status('processing', 93, save_msg)
                
                # Attempt to detect ship regions
                detect_msg = "Detecting ship regions based on vibration energy"
                append_to_detailed_log(detect_msg)
                update_processing_status('processing', 93, detect_msg)
                
                try:
                    # Use the detector class directly instead of calling estimator.detect_ship_regions
                    detector_regions = detector.detect_regions(estimator.vibration_energy_map_db)
                    estimator.ship_regions = detector_regions
                    
                    if detector_regions and len(detector_regions) > 0:
                        ship_regions_file = os.path.join(result_dir, 'ship_regions.png')
                        # If a plot_ship_regions method exists, use it
                        if hasattr(estimator, 'plot_ship_regions'):
                            estimator.plot_ship_regions(output_file=ship_regions_file)
                            regions_msg = f"Saved ship regions visualization to {ship_regions_file}"
                            append_to_detailed_log(regions_msg)
                            update_processing_status('processing', 94, regions_msg)
                        
                        detect_done_msg = f"Detected {len(detector_regions)} ship regions"
                        append_to_detailed_log(detect_done_msg)
                        update_processing_status('processing', 94, detect_done_msg)
                    else:
                        no_regions_msg = "No ship regions detected above threshold"
                        append_to_detailed_log(no_regions_msg)
                        update_processing_status('processing', 94, no_regions_msg)
                except Exception as e:
                    error_msg = f"Error detecting ship regions: {str(e)}"
                    append_to_detailed_log(error_msg)
                    update_processing_status('processing', 94, error_msg)
            else:
                fail_msg = "Failed to calculate vibration energy map"
                append_to_detailed_log(fail_msg)
                update_processing_status('processing', 93, fail_msg)
        except Exception as e:
            error_msg = f"Error saving vibration energy map: {str(e)}"
            append_to_detailed_log(error_msg)
            update_processing_status('processing', 93, error_msg)
        
        # Save individual measurement point results
        for i in range(len(measurement_points)):
            try:
                point_result_file = os.path.join(result_dir, f'point_{i}_results.png')
                estimator.plot_results(i, output_file=point_result_file)
                point_msg = f"Saved point {i} results to {point_result_file}"
                append_to_detailed_log(point_msg)
                update_processing_status('saving', 96 + i, point_msg)
            except Exception as e:
                error_msg = f"Error saving point {i} results: {str(e)}"
                append_to_detailed_log(error_msg)
                update_processing_status('saving', 96 + i, error_msg)
        
        # Save metadata about the processing
        metadata = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                'data_type': data_type,
                'num_subapertures': num_subapertures,
                'window_size': window_size,
                'overlap': overlap
            },
            'file': os.path.basename(file_path) if not use_demo else 'demo',
            'measurement_points': [list(mp) for mp in measurement_points],
            'processing_time': datetime.datetime.now().strftime("%H:%M:%S")
        }
        
        with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a human-readable summary
        with open(os.path.join(result_dir, 'summary.txt'), 'w') as f:
            f.write(f"Processing completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {os.path.basename(file_path) if not use_demo else 'demo'}\n")
            f.write(f"Data type: {data_type.upper()}\n")
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
        
        # Create the final completion message
        completion_msg = f'Processing completed. Results saved to {os.path.basename(result_dir)}'
        # Log it
        append_to_detailed_log(completion_msg)
        
        # Force update the UI status to complete with 100% progress
        # This should override any intermediate status set by the callback
        session['processing_status'] = {
            'status': 'complete',
            'progress': 100,
            'details': completion_msg,
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
            'log_messages': session.get('processing_status', {}).get('log_messages', [])
        }
        session.modified = True
        if hasattr(session, 'save_session'):
            session.save_session(None)  # Force session save
        
        # Update UI status through the regular function as well
        update_processing_status('complete', 100, completion_msg, console_log="Processing completed successfully")
        flash(f'Processing completed successfully. Results saved to {os.path.basename(result_dir)}', 'success')
        
    except Exception as e:
        logger.exception(f"Unexpected error during processing: {str(e)}")
        
        # Write the exception to the detailed log
        try:
            error_details = f"\n\nERROR: {str(e)}"
            append_to_detailed_log(error_details)
            
            import traceback
            trace = traceback.format_exc()
            append_to_detailed_log(trace)
            
            # Also write to a dedicated error file
            with open(os.path.join(result_dir, 'error_details.txt'), 'w') as f:
                f.write(f"Error during processing: {str(e)}\n\n")
                f.write(trace)
        except Exception as log_err:
            logger.error(f"Error writing exception details: {log_err}")
        
        # Log the error through our helper
        error_msg = f'Error during processing: {str(e)}'
        append_to_detailed_log(error_msg)
        
        # Force error status in session
        session['processing_status'] = {
            'status': 'error',
            'progress': 100,
            'details': error_msg,
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
            'log_messages': session.get('processing_status', {}).get('log_messages', [])
        }
        session.modified = True
        if hasattr(session, 'save_session'):
            session.save_session(None)  # Force session save
        
        # Update UI status through regular function
        update_processing_status('error', 100, error_msg)
        flash(error_msg, 'error')
    
    finally:
        # Remove the file handler
        logger.removeHandler(file_handler)
        
        # Write final status to the detailed log
        try:
            final_log_msg = f"\n=== Processing finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
            append_to_detailed_log(final_log_msg)
        except Exception as e:
            logger.error(f"Error writing final log message: {e}")
    
    # At the very end of the function, after all processing
    return redirect(url_for('index'))

@app.route('/results/<path:filename>')
def results_file(filename):
    """Serve a file from the results directory"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# WebSocket routes
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    logger.info(f"Client connected to WebSocket: {client_id}")
    # Send current processing status on connection
    if 'processing_status' in session:
        emit('status_update', session['processing_status'])
    else:
        # Send default idle status
        emit('status_update', {
            'status': 'idle',
            'progress': 0,
            'details': 'Ready to process',
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
            'log_messages': []
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    logger.info(f"Client disconnected from WebSocket: {client_id}")

@socketio.on('get_current_status')
def handle_get_status():
    """Send current status on request"""
    client_id = request.sid
    logger.debug(f"Status requested by client: {client_id}")
    
    if 'processing_status' in session:
        logger.debug(f"Sending current status: {session['processing_status'].get('status', 'unknown')}")
        emit('status_update', session['processing_status'])
    else:
        logger.debug("No current status in session, sending idle")
        emit('status_update', {
            'status': 'idle',
            'progress': 0,
            'details': 'Ready to process',
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
            'log_messages': []
        })

# Update the main app to run with SocketIO
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5002)
