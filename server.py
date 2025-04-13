from flask import Flask, request, jsonify, send_file, after_this_request
import os
import tempfile
from werkzeug.utils import secure_filename
from main import process_document, process_text, play_audio  # Import your existing functions
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech and return the audio file."""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text parameter'}), 400
    
    try:
        # Create a temporary file for the audio
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"tts_{os.urandom(8).hex()}.wav")
        audio_file = process_text(data['text'], output_file=temp_file)
        
        @after_this_request
        def cleanup(response):
            try:
                os.remove(audio_file)
            except Exception as e:
                app.logger.error(f"Error removing temporary file {audio_file}: {e}")
            return response
        
        return send_file(audio_file, mimetype='audio/wav', as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/document/tts', methods=['POST'])
def document_to_speech():
    """Convert uploaded document to speech."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_input = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_input)
            
            # Process the document
            audio_file = process_document(temp_input)
            
            if not audio_file or not os.path.exists(audio_file):
                return jsonify({'error': 'Failed to process document'}), 500
            
            @after_this_request
            def cleanup(response):
                try:
                    os.remove(temp_input)
                    os.remove(audio_file)
                except Exception as e:
                    app.logger.error(f"Error removing temporary files: {e}")
                return response
            
            return send_file(audio_file, mimetype='audio/wav', as_attachment=True)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/play', methods=['POST'])
def play_audio_endpoint():
    """Play audio file from path (for local testing)."""
    data = request.json
    if not data or 'filepath' not in data:
        return jsonify({'error': 'Missing filepath parameter'}), 400
    
    try:
        play_audio(data['filepath'])
        return jsonify({'status': 'played successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(host='0.0.0.0', port=5000, debug=True)