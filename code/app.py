from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
from micromotion_estimator import ShipMicroMotionEstimator
from ship_region_detector import ShipRegionDetector
from flask_session import Session  # Import Flask-Session

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

@app.route('/')
def index():
    uploaded_files = list_uploaded_files()
    return render_template('index.html', uploaded_files=uploaded_files)

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
    if step == 'load_data' or not hasattr(estimator, 'data_loaded') or not estimator.data_loaded:
        if use_demo:
            # Use synthetic data for demo
            if estimator.load_data('demo'):
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
            else:
                error = 'Error loading synthetic data'
        else:
            # Use real data
            if not filename:
                error = 'No file selected'
            else:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if not os.path.exists(filepath):
                    error = f'File {filename} not found'
                else:
                    if estimator.load_data(filepath):
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
                    else:
                        error = f'Error loading file {filename}'
    
    # Run subaperture creation step
    if step == 'create_subapertures' and not error:
        if not hasattr(estimator, 'data_loaded') or not estimator.data_loaded:
            error = 'Data must be loaded first'
        else:
            if hasattr(estimator, 'create_subapertures') and callable(getattr(estimator, 'create_subapertures')):
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
            else:
                # If the method doesn't exist, show a message
                error = 'create_subapertures method not available'
    
    # Run focus subapertures step
    if step == 'focus_subapertures' and not error:
        if not hasattr(estimator, 'subapertures') or not estimator.subapertures:
            error = 'Subapertures must be created first'
        else:
            if hasattr(estimator, 'focus_subapertures') and callable(getattr(estimator, 'focus_subapertures')):
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
                        results['slc_images_plot'] = fig_to_base64(fig)
                        plt.close(fig)
                else:
                    error = 'Error focusing subapertures'
            else:
                error = 'focus_subapertures method not available'
    
    # Run displacement estimation step
    if step == 'estimate_displacement' and not error:
        if not hasattr(estimator, 'slc_images') or not estimator.slc_images:
            error = 'SLC images must be created first'
        else:
            if estimator.estimate_displacement():
                results['message'] = 'Displacement estimated successfully'
                # Visualize the displacement maps
                if hasattr(estimator, 'displacement_maps') and estimator.displacement_maps:
                    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Plot range displacement
                    range_disp = estimator.displacement_maps.get('range', {}).get(0, None)
                    if range_disp is not None:
                        im1 = axs[0, 0].imshow(range_disp, cmap='coolwarm', 
                                             vmin=-1, vmax=1, aspect='auto')
                        axs[0, 0].set_title('Range Displacement Map (frame 0)')
                        plt.colorbar(im1, ax=axs[0, 0], label='Displacement (pixels)')
                    
                    # Plot azimuth displacement
                    azimuth_disp = estimator.displacement_maps.get('azimuth', {}).get(0, None)
                    if azimuth_disp is not None:
                        im2 = axs[0, 1].imshow(azimuth_disp, cmap='coolwarm', 
                                             vmin=-1, vmax=1, aspect='auto')
                        axs[0, 1].set_title('Azimuth Displacement Map (frame 0)')
                        plt.colorbar(im2, ax=axs[0, 1], label='Displacement (pixels)')
                    
                    # Plot SNR
                    if hasattr(estimator, 'snr_maps') and estimator.snr_maps:
                        snr_map = estimator.snr_maps.get(0, None)
                        if snr_map is not None:
                            im3 = axs[1, 0].imshow(snr_map, cmap='viridis', 
                                                 vmin=0, vmax=np.percentile(snr_map, 95), aspect='auto')
                            axs[1, 0].set_title('SNR Map (frame 0)')
                            plt.colorbar(im3, ax=axs[1, 0], label='SNR')
                    
                    # Plot phase coherence
                    if hasattr(estimator, 'coherence_maps') and estimator.coherence_maps:
                        coherence_map = estimator.coherence_maps.get(0, None)
                        if coherence_map is not None:
                            im4 = axs[1, 1].imshow(coherence_map, cmap='viridis', 
                                                 vmin=0, vmax=1, aspect='auto')
                            axs[1, 1].set_title('Phase Coherence Map (frame 0)')
                            plt.colorbar(im4, ax=axs[1, 1], label='Coherence')
                    
                    plt.tight_layout()
                    results['displacement_plot'] = fig_to_base64(fig)
                    plt.close(fig)
            else:
                error = 'Error estimating displacement'
    
    # Run time series analysis step
    if step == 'analyze_time_series' and not error:
        if not hasattr(estimator, 'displacement_maps') or not estimator.displacement_maps:
            error = 'Displacement maps must be created first'
        else:
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
    
    # Run vibration energy calculation step
    if step == 'calculate_vibration_energy' and not error:
        if not hasattr(estimator, 'time_series') or not estimator.time_series:
            error = 'Time series must be analyzed first'
        else:
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
    
    # Run ship region detection step
    if step == 'detect_ship_regions' and not error:
        if not hasattr(estimator, 'vibration_energy_map_db') or estimator.vibration_energy_map_db is None:
            error = 'Vibration energy map must be calculated first'
        else:
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
    # Get form parameters
    num_subapertures = int(request.form.get('num_subapertures', 7))
    window_size = int(request.form.get('window_size', 64))
    overlap = float(request.form.get('overlap', 0.5))
    use_demo = request.form.get('use_demo', 'false') == 'true'
    
    # Create estimator
    estimator = ShipMicroMotionEstimator(
        num_subapertures=num_subapertures,
        window_size=window_size,
        overlap=overlap
    )
    
    # Process data
    if use_demo:
        # Use synthetic data for demo
        if estimator.load_data('demo'):
            flash('Using synthetic data for demonstration', 'info')
        else:
            flash('Error creating synthetic data', 'error')
            return redirect(url_for('index'))
    else:
        # Check if a file was uploaded or an existing file was selected
        existing_file = request.form.get('existing_file', '')
        
        if existing_file:
            # Use existing file
            filename = secure_filename(existing_file)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(filepath):
                flash(f'Selected file {filename} not found', 'error')
                return redirect(url_for('index'))
                
            flash(f'Using existing file: {filename}', 'info')
        else:
            # Check for new file upload
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(url_for('index'))
            
            file = request.files['file']
            
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            if not allowed_file(file.filename):
                flash(f'File type not allowed. Please upload a CPHD file.', 'error')
                return redirect(url_for('index'))
            
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Check if file already exists in the uploads directory
            if os.path.exists(filepath):
                flash(f'File {filename} already exists, using the existing file', 'info')
            else:
                # Save the file only if it doesn't already exist
                file.save(filepath)
                flash(f'File {filename} uploaded successfully', 'info')
        
        # Load the data
        if not estimator.load_data(filepath):
            flash('Error loading data', 'error')
            return redirect(url_for('index'))
    
    # Estimate displacement
    if not estimator.estimate_displacement():
        flash('Error estimating displacement', 'error')
        return redirect(url_for('index'))
    
    # Define measurement points (for demo, use fixed points)
    if use_demo:
        measurement_points = [(150, 200), (300, 350)]
    else:
        # For real data, we would need to detect ships and select points
        # This is a placeholder
        measurement_points = [(100, 100), (200, 200)]
    
    # Analyze time series
    if not estimator.analyze_time_series(measurement_points):
        flash('Error analyzing time series', 'error')
        return redirect(url_for('index'))
    
    # Calculate vibration energy map
    if not estimator.calculate_vibration_energy_map():
        flash('Error calculating vibration energy map', 'error')
        return redirect(url_for('index'))
    
    # Detect ship regions
    if not estimator.detect_ship_regions(num_regions=3, energy_threshold=-15):
        flash('Error detecting ship regions', 'error')
        # Continue anyway, as this is not critical
    
    # Use the ShipRegionDetector for more advanced detection
    detector = ShipRegionDetector(min_region_size=20, energy_threshold=-15, num_regions=3)
    ship_regions = detector.detect_regions(estimator.vibration_energy_map_db)
    
    # Get region statistics
    region_stats = []
    if ship_regions:
        region_stats = detector.get_region_statistics(
            estimator.vibration_energy_map_db, estimator.displacement_maps
        )
    
    # Generate plots
    results = {}
    
    # Generate time series and frequency spectra plots for each measurement point
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
        
        # Convert to base64 for HTML display
        results[f'point_{i}_plot'] = fig_to_base64(fig)
        plt.close(fig)
    
    # Generate vibration energy map visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot SLC image on the left
    if estimator.slc_images is not None and len(estimator.slc_images) > 0:
        # Use the first SLC image
        slc_image = estimator.slc_images[0]
        axs[0].imshow(slc_image, cmap='gray')
        axs[0].set_title('SLC Image with Ship Regions')
        axs[0].set_xlabel('Range (pixels)')
        axs[0].set_ylabel('Azimuth (pixels)')
    
    # Plot vibration energy map on the right
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=-25, vmax=0)
    im = axs[1].imshow(estimator.vibration_energy_map_db, cmap=cmap, norm=norm)
    axs[1].set_title('SLC ROI Vibration Energy (dB)')
    axs[1].set_xlabel('Range (pixels)')
    axs[1].set_ylabel('Azimuth (pixels)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axs[1])
    cbar.set_label('Vibration Energy (dB)')
    
    # Add ship region labels and annotations
    if ship_regions:
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
            arrow_angle = np.random.uniform(0, 2*np.pi)  # Random angle for variety
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
    
    # Convert to base64 for HTML display
    results['vibration_energy_map'] = fig_to_base64(fig)
    plt.close(fig)
    
    # Identify dominant frequencies
    dominant_frequencies = []
    for i in range(len(measurement_points)):
        freqs = estimator.identify_dominant_frequencies(i)
        if freqs:
            dominant_frequencies.append({
                'point': i,
                'range': [{'frequency': f, 'amplitude': a} for f, a in freqs['range']],
                'azimuth': [{'frequency': f, 'amplitude': a} for f, a in freqs['azimuth']]
            })
    
    # Save results to file
    vibration_energy_map_path = os.path.join(app.config['RESULTS_FOLDER'], 'vibration_energy_map.png')
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot SLC image on the left
    if estimator.slc_images is not None and len(estimator.slc_images) > 0:
        axs[0].imshow(estimator.slc_images[0], cmap='gray')
        axs[0].set_title('SLC Image with Ship Regions')
        axs[0].set_xlabel('Range (pixels)')
        axs[0].set_ylabel('Azimuth (pixels)')
    
    # Plot vibration energy map on the right
    im = axs[1].imshow(estimator.vibration_energy_map_db, cmap=cmap, norm=norm)
    axs[1].set_title('SLC ROI Vibration Energy (dB)')
    axs[1].set_xlabel('Range (pixels)')
    axs[1].set_ylabel('Azimuth (pixels)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axs[1])
    cbar.set_label('Vibration Energy (dB)')
    
    # Add ship region labels and annotations
    if ship_regions:
        for region in ship_regions:
            region_id = region['id']
            centroid = region['centroid']
            
            # Add label to both plots
            axs[0].text(centroid[1], centroid[0], str(region_id), 
                       color='white', fontsize=12, ha='center', va='center',
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
            
            axs[1].text(centroid[1], centroid[0], str(region_id), 
                       color='white', fontsize=12, ha='center', va='center',
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Add arrows
            arrow_length = 40
            arrow_angle = np.random.uniform(0, 2*np.pi)
            arrow_dx = arrow_length * np.cos(arrow_angle)
            arrow_dy = arrow_length * np.sin(arrow_angle)
            arrow_start_x = centroid[1] + arrow_dx
            arrow_start_y = centroid[0] + arrow_dy
            
            axs[0].annotate('', xy=(centroid[1], centroid[0]), 
                           xytext=(arrow_start_x, arrow_start_y),
                           arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
            
            axs[1].annotate('', xy=(centroid[1], centroid[0]), 
                           xytext=(arrow_start_x, arrow_start_y),
                           arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8))
    
    plt.tight_layout()
    plt.savefig(vibration_energy_map_path)
    plt.close(fig)
    
    # Render results template
    return render_template(
        'results.html',
        results=results,
        dominant_frequencies=dominant_frequencies,
        region_stats=region_stats,
        num_points=len(measurement_points)
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
