import os
import datetime
import json
import time
from flask import Flask, request, jsonify
import whisperx
from werkzeug.utils import secure_filename
from typing import Dict, Any
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configure TF32 and suppress warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Initialize Qdrant client
qdrant = QdrantClient(os.getenv('QDRANT_URL', 'http://localhost:6333'))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant collection with retries
max_retries = 5
retry_delay = 2  # seconds

for attempt in range(max_retries):
    try:
        # Check if collection exists
        try:
            qdrant.get_collection('transcriptions')
        except Exception:
            # Create collection if it doesn't exist
            qdrant.create_collection(
                collection_name='transcriptions',
                vectors_config={
                    'size': 384,  # all-MiniLM-L6-v2 embedding size
                    'distance': 'Cosine'
                }
            )
        break
    except Exception as e:
        if attempt == max_retries - 1:
            app.logger.error(f"Failed to initialize Qdrant collection after {max_retries} attempts: {str(e)}")
            raise
        app.logger.warning(f"Qdrant initialization attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)

app = Flask(__name__)

# Configuration
MODEL_NAME = os.getenv('WHISPER_MODEL', 'large-v2')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Verify GPU availability
import torch
if torch.cuda.is_available():
    app.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    app.logger.info(f"CUDA version: {torch.version.cuda}")
    device = 'cuda'
else:
    app.logger.warning("No GPU available, falling back to CPU")
    device = 'cpu'

# Load model at startup
model = whisperx.load_model(MODEL_NAME, device=device)

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
        # Secure filename and save
        filename = secure_filename(file.filename)
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        app.logger.info(f"Starting transcription for file: {filename}")
        
        # Initial transcription
        try:
            result = model.transcribe(temp_path, batch_size=16)
            app.logger.info(f"Initial transcription complete. Language detected: {result.get('language', 'unknown')}")
            
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict result, got {type(result)}")
            if 'segments' not in result:
                raise ValueError(f"No segments in result. Keys present: {result.keys()}")
        except Exception as e:
            app.logger.error(f"Transcription failed: {str(e)}")
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

        # Alignment
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                temp_path,
                device,
                return_char_alignments=False
            )
            app.logger.info("Alignment completed successfully")
        except Exception as e:
            app.logger.error(f"Alignment failed: {str(e)}")
            # Continue with unaligned result rather than failing
            pass

        # Diarization
        try:
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment variables")

            min_speakers = int(os.getenv('DIARIZATION_MIN_SPEAKERS', 1))
            max_speakers = int(os.getenv('DIARIZATION_MAX_SPEAKERS', 5))
            
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
            
            diarize_segments = diarize_model(
                temp_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Assign speakers
            result = whisperx.assign_word_speakers(diarize_segments, result)
            app.logger.info("Diarization completed successfully")
            
        except Exception as e:
            app.logger.error(f"Diarization failed: {str(e)}")
            # Fallback to single speaker
            for segment in result['segments']:
                segment['speaker'] = 'SPEAKER_00'
            app.logger.info("Using fallback single speaker mode")

        # Process segments and create response
        try:
            # Ensure all segments have required fields
            processed_segments = []
            for segment in result['segments']:
                processed_segment = {
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'text': segment.get('text', '').strip(),
                    'speaker': segment.get('speaker', 'SPEAKER_00')
                }
                processed_segments.append(processed_segment)

            # Create full text with speaker annotations
            text = ' '.join(
                f"[Speaker {seg['speaker']}] {seg['text']}"
                for seg in processed_segments
            )

            response = {
                'transcription': text,
                'language': result.get('language', 'en'),
                'segments': processed_segments
            }

            # Store in Qdrant
            try:
                embedding = embedder.encode(text)
                qdrant.upsert(
                    collection_name='transcriptions',
                    points=[{
                        'id': int(datetime.datetime.now().timestamp() * 1000),
                        'vector': embedding.tolist(),
                        'payload': {
                            'text': text,
                            'language': response['language'],
                            'segments': processed_segments,
                            'filename': filename,
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                    }]
                )
                app.logger.info("Successfully stored in Qdrant")
            except Exception as e:
                app.logger.error(f"Qdrant storage failed: {str(e)}")
                # Continue without failing the request

            # Clean up
            os.remove(temp_path)
            
            return jsonify(response)

        except Exception as e:
            app.logger.error(f"Error processing segments: {str(e)}")
            return jsonify({'error': f'Failed to process segments: {str(e)}'}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        # Ensure temp file is cleaned up even if processing fails
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 2990))
    app.run(host='0.0.0.0', port=port)
