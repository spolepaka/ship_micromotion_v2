from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
from micromotion_estimator import ShipMicroMotionEstimator
from ship_region_detector import ShipRegionDetector

app = Flask(__name__)
app.secret_key = 'ship_micromotion_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.getcwd(), 'results')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5 GB max upload size

# Create upload and results folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'cphd', 'h5', 'hdf5'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

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
        # Check if a file was uploaded
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
        file.save(filepath)
        
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
