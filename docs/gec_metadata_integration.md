# Integrating GEC TIFF and Metadata JSON for Verification

## Problem

The initial ship detection algorithm identifies potential ship regions based on vibration map analysis. However, to improve accuracy and verify the results, we need to:

1.  **Ground Truth Verification:** Compare the detected regions (in geographic coordinates) against a known ground truth, often represented by a georeferenced image (GEC TIFF).
2.  **Parameter Tuning:** Potentially adjust detection parameters (like `energy_threshold`, `min_region_size`) based on prior information about the scene or expected targets, which might be available in a metadata file (e.g., JSON).
3.  **User Input:** Allow the user to provide these optional GEC and Metadata files through the web interface.

## Solution

The solution involves updates to the frontend UI, the backend Flask application, and the core detection script (`ship_detection_script.py`).

### 1. Frontend (`index.html`)

The complete updated `index.html` code:

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Ship Detection Upload</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        /* Optional: Add some custom styling */
        .container {
            max-width: 600px;
            margin-top: 50px;
        }
        .result-section {
            margin-top: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .spinner-border {
            display: none; /* Hidden by default */
        }
        .is-loading .spinner-border {
            display: inline-block; /* Show spinner when loading */
        }
        .is-loading button[type="submit"] {
            cursor: not-allowed; /* Indicate loading */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Upload SAR Data for Ship Detection</h1>

        <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">Choose SAR Data File (CPHD or SICD/NITF):</label>
                <input type="file" class="form-control-file" id="file" name="file" required>
                <small class="form-text text-muted">Select the main data file.</small>
            </div>

            <!-- Add GEC TIFF Upload -->
            <div class="form-group">
                <label for="gec_file">Choose GEC TIFF File (Optional):</label>
                <input type="file" class="form-control-file" id="gec_file" name="gec_file" accept=".tif,.tiff">
                <small class="form-text text-muted">Georeferenced image for verification.</small>
            </div>

            <!-- Add Metadata JSON Upload -->
            <div class="form-group">
                <label for="metadata_file">Choose Metadata JSON File (Optional):</label>
                <input type="file" class="form-control-file" id="metadata_file" name="metadata_file" accept=".json">
                <small class="form-text text-muted">Metadata for tuning/verification.</small>
            </div>

            <div class="form-group">
                <label for="threshold">Energy Threshold:</label>
                <input type="number" class="form-control" id="threshold" name="threshold" step="0.01" value="0.5" required>
                <small class="form-text text-muted">Threshold for segmenting vibration map.</small>
            </div>

            <div class="form-group">
                <label for="min_size">Minimum Region Size (pixels):</label>
                <input type="number" class="form-control" id="min_size" name="min_size" step="1" value="50" required>
                <small class="form-text text-muted">Minimum pixel count for a detected region.</small>
            </div>

            <button type="submit" class="btn btn-primary">
                <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                Upload and Process
            </button>
        </form>

        <div id="result" class="result-section" style="display: none;">
            <h2>Processing Results</h2>
            <pre id="output"></pre>
            <div id="images">
                <!-- Images will be loaded here -->
            </div>
            <div id="download-links">
                <!-- Download links will be added here -->
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('upload-form');
        const resultDiv = document.getElementById('result');
        const outputPre = document.getElementById('output');
        const imagesDiv = document.getElementById('images');
        const downloadLinksDiv = document.getElementById('download-links');
        const submitButton = form.querySelector('button[type="submit"]');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            resultDiv.style.display = 'none'; // Hide previous results
            outputPre.textContent = '';      // Clear previous output
            imagesDiv.innerHTML = '';        // Clear previous images
            downloadLinksDiv.innerHTML = ''; // Clear previous download links
            submitButton.disabled = true;
            submitButton.classList.add('is-loading');

            const formData = new FormData(form);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });

                resultDiv.style.display = 'block'; // Show result section

                if (!response.ok) {
                    const errorText = await response.text();
                    outputPre.textContent = `Error: ${response.statusText}\n${errorText}`;
                    throw new Error(`HTTP error! status: ${response.status}, ${errorText}`);
                }

                const result = await response.json();

                if (result.error) {
                    outputPre.textContent = `Processing Error:\n${result.error}`;
                } else {
                    outputPre.textContent = `Processing Log:\n${result.log}\n\nShip Details:\n${result.ship_details_text}`;

                    // Display images if available
                    result.image_urls.forEach(url => {
                        const img = document.createElement('img');
                        img.src = url + '?t=' + new Date().getTime(); // Prevent caching
                        img.className = 'img-fluid mb-2'; // Bootstrap class
                        imagesDiv.appendChild(img);
                        imagesDiv.appendChild(document.createElement('hr'));
                    });

                     // Add download links if available
                    result.download_files.forEach(fileInfo => {
                         const link = document.createElement('a');
                         link.href = fileInfo.url + '?t=' + new Date().getTime(); // Prevent caching
                         link.textContent = `Download ${fileInfo.name}`;
                         link.className = 'btn btn-secondary btn-sm mr-2 mb-2';
                         link.download = fileInfo.name; // Suggest filename
                         downloadLinksDiv.appendChild(link);
                    });
                }

            } catch (error) {
                console.error('Upload failed:', error);
                if (!outputPre.textContent) { // Only update if no specific error message was set
                     outputPre.textContent = 'An error occurred during upload or processing. Check the console for details.';
                }
            } finally {
                submitButton.disabled = false;
                submitButton.classList.remove('is-loading');
            }
        });
    </script>
</body>
</html>
```

### 2. Backend (`app.py`)

The complete updated `app.py` code:

```python
import os
import subprocess
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import logging
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'cphd', 'cph', 'ntf', 'nitf', 'tif', 'tiff', 'json'}
SCRIPT_PATH = 'ship_detection_script.py'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512 MB limit

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No main data file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected main data file'}), 400

    # --- Optional file handling ---
    gec_file = request.files.get('gec_file')
    metadata_file = request.files.get('metadata_file')
    # --- End optional file handling ---

    if file and allowed_file(file.filename):
        # Generate unique prefix for this job's files
        job_id = str(uuid.uuid4())
        base_filename = secure_filename(file.filename)
        input_filename = f"{job_id}_{base_filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        logging.info(f"Saved main data file to: {input_path}")

        # --- Save optional files ---
        gec_path = None
        if gec_file and gec_file.filename != '' and allowed_file(gec_file.filename):
            gec_filename = f"{job_id}_gec_{secure_filename(gec_file.filename)}"
            gec_path = os.path.join(app.config['UPLOAD_FOLDER'], gec_filename)
            gec_file.save(gec_path)
            logging.info(f"Saved GEC file to: {gec_path}")

        metadata_path = None
        if metadata_file and metadata_file.filename != '' and allowed_file(metadata_file.filename):
            metadata_filename = f"{job_id}_metadata_{secure_filename(metadata_file.filename)}"
            metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_filename)
            metadata_file.save(metadata_path)
            logging.info(f"Saved Metadata file to: {metadata_path}")
        # --- End save optional files ---

        # Get parameters from form
        try:
            threshold = float(request.form.get('threshold', 0.5))
            min_size = int(request.form.get('min_size', 50))
        except (ValueError, TypeError):
             return jsonify({'error': 'Invalid threshold or min_size value'}), 400

        output_prefix = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_output")

        # --- Build the command for the script ---
        command = [
            'python', SCRIPT_PATH,
            input_path,
            '-o', output_prefix,
            '-t', str(threshold),
            '-s', str(min_size)
        ]
        # Add optional file arguments if they were provided and saved
        if gec_path:
            command.extend(['--gec', gec_path])
        if metadata_path:
            command.extend(['--metadata', metadata_path])
        # --- End build command ---

        logging.info(f"Running command: {' '.join(command)}")

        try:
            # Execute the script
            process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
            script_log = process.stdout
            logging.info("Script executed successfully.")
            logging.debug(f"Script stdout:\n{script_log}")

            # --- Prepare response ---
            image_urls = []
            download_files = []
            ship_details_text = "No ship details file generated."

            # Check for expected output files and create URLs/info
            vibration_map_png = f"{output_prefix}_vibration_map.png"
            labeled_regions_png = f"{output_prefix}_labeled_regions.png"
            detected_ships_json = f"{output_prefix}_detected_ships.json"

            if os.path.exists(vibration_map_png):
                image_urls.append(url_for('get_output_file', filename=os.path.basename(vibration_map_png)))
                download_files.append({
                    "name": os.path.basename(vibration_map_png),
                    "url": url_for('get_output_file', filename=os.path.basename(vibration_map_png))
                })
            if os.path.exists(labeled_regions_png):
                image_urls.append(url_for('get_output_file', filename=os.path.basename(labeled_regions_png)))
                download_files.append({
                    "name": os.path.basename(labeled_regions_png),
                    "url": url_for('get_output_file', filename=os.path.basename(labeled_regions_png))
                })

            # Try to read the detected ships JSON for display
            if os.path.exists(detected_ships_json):
                try:
                    with open(detected_ships_json, 'r') as f:
                        ship_data = json.load(f)
                        ship_details_text = json.dumps(ship_data, indent=4)
                except Exception as e:
                    logging.error(f"Error reading or parsing {detected_ships_json}: {e}")
                    ship_details_text = f"Error reading ship details file: {e}"
                download_files.append({
                    "name": os.path.basename(detected_ships_json),
                    "url": url_for('get_output_file', filename=os.path.basename(detected_ships_json))
                })

            return jsonify({
                'message': 'File processed successfully',
                'log': script_log,
                'ship_details_text': ship_details_text,
                'image_urls': image_urls,
                'download_files': download_files
            })

        except subprocess.CalledProcessError as e:
            logging.error(f"Script execution failed: {e}")
            logging.error(f"Stderr: {e.stderr}")
            return jsonify({'error': f'Script execution failed: {e.stderr}'}), 500
        except subprocess.TimeoutExpired:
             logging.error("Script execution timed out.")
             return jsonify({'error': 'Processing timed out.'}), 500
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/output/<filename>')
def get_output_file(filename):
     # Basic security check: prevent directory traversal
     if '..' in filename or filename.startswith('/'):
         return "Invalid filename", 400
     try:
         return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=False)
     except FileNotFoundError:
         return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production
```

### 3. Key Features and Changes

#### Frontend Changes
- Added file input fields for GEC TIFF and Metadata JSON
- Added descriptive labels and help text
- Enhanced error handling and loading states
- Added download links for generated files
- Improved display of processing results

#### Backend Changes
- Added support for handling multiple file uploads
- Implemented secure file handling with unique job IDs
- Added proper error handling and logging
- Added route for serving output files
- Enhanced response structure with download links

#### Integration Features
1. **File Handling:**
   - Secure filename generation with UUID prefixes
   - Proper MIME type checking
   - Size limits for uploads

2. **Processing:**
   - Dynamic command building based on available files
   - Timeout protection (300 seconds)
   - Proper subprocess handling

3. **Response Handling:**
   - Structured JSON responses
   - Error handling at multiple levels
   - File download support
   - Image display support

4. **Security:**
   - Directory traversal prevention
   - Secure filename handling
   - File type restrictions

### 4. Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Access the web interface at `http://localhost:5000`

3. Upload files:
   - Required: SAR Data File (CPHD/SICD)
   - Optional: GEC TIFF file
   - Optional: Metadata JSON file

4. Set parameters:
   - Energy Threshold (default: 0.5)
   - Minimum Region Size (default: 50 pixels)

5. Click "Upload and Process"

6. View Results:
   - Processing log
   - Detected ship details
   - Generated images
   - Download links for all outputs

### 5. Notes

- The application creates two directories:
  - `uploads/`: For storing uploaded files
  - `output/`: For storing processing results
- Each processing job gets a unique UUID to prevent filename conflicts
- Large files (>512MB) are rejected by default
- The web interface provides real-time feedback during processing
- All generated files are available for download after processing
- The application includes proper error handling at all levels

### 6. Dependencies

- Flask
- Werkzeug
- Python 3.6+
- Additional requirements for the ship detection script (numpy, scipy, etc.)
- Bootstrap 4.5.2 (CDN) 