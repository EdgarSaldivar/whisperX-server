import os
from flask import Flask, request, jsonify
import whisperx
from werkzeug.utils import secure_filename
from typing import Dict, Any

app = Flask(__name__)

# Configuration
MODEL_NAME = os.getenv('WHISPER_MODEL', 'large-v2')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Load model at startup
model = whisperx.load_model(MODEL_NAME, device='cuda')

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
def transcribe() -> Dict[str, Any]:
    # Check if file is present
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    # Validate file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400
    
    if file.content_length > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large'}), 413
    
    try:
        # Secure filename and process
        filename = secure_filename(file.filename)
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        # Transcribe audio
        result = model.transcribe(temp_path)
        
        # Handle WhisperX response format
        if isinstance(result, dict) and 'segments' in result:
            text = ' '.join(segment['text'].strip() for segment in result['segments'])
            response = {
                'transcription': text,
                'language': result.get('language', 'en'),
                'segments': result['segments']
            }
            
            # Save transcription to volume
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{os.getenv('TRANSCRIPTION_DIR', '/transcriptions')}/{timestamp}_{secure_filename(file.filename)}.json"
            with open(filename, 'w') as f:
                json.dump(response, f)
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify(response)
        else:
            # Clean up
            os.remove(temp_path)
            return jsonify({'error': 'Invalid transcription result format'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 2990))
    app.run(host='0.0.0.0', port=port)
