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
        # Secure filename and process
        filename = secure_filename(file.filename)
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        # Transcribe audio
        result = model.transcribe(temp_path)
        
        # Handle WhisperX response format with diarization
        if isinstance(result, dict) and 'segments' in result:
            # Perform diarization with WhisperX pipeline
            try:
                hf_token = os.getenv('HF_TOKEN')
                if not hf_token:
                    raise ValueError("HF_TOKEN not found in environment variables")
                    
                app.logger.info("Initializing WhisperX diarization pipeline...")
                
                # Transcribe with WhisperX
                result = whisperx.transcribe(
                    audio=temp_path,
                    model=model,
                    device=device,
                    batch_size=16,
                    compute_type="float16"
                )
                
                # Align output
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device=device
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    temp_path,
                    device,
                    return_char_alignments=False
                )
                
                # Diarize
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                diarize_segments = diarize_model(temp_path)
                
                # Assign speakers
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                app.logger.info("Diarization completed successfully")
                
                # Use the WhisperX assigned speaker segments
                diarize_segments = result['segments']
            except Exception as e:
                app.logger.error(f"Diarization failed: {str(e)}")
                diarize_segments = result['segments']
                for segment in diarize_segments:
                    segment['speaker'] = 'SPEAKER_00'
            
            # Combine text with speaker information
            text = ' '.join(f"[Speaker {segment['speaker']}] {segment['text'].strip()}"
                          for segment in result['segments'])
            
            # Align diarization results with transcription segments
            for segment in result['segments']:
                # Find overlapping speakers
                speakers = []
                for turn in diarize_segments:
                    try:
                        turn_start = turn[0] if isinstance(turn, (list, tuple)) else turn['start']
                        turn_end = turn[1] if isinstance(turn, (list, tuple)) else turn['end']
                        speaker = turn[2] if isinstance(turn, (list, tuple)) else turn['speaker']
                        
                        if turn_start <= segment['end'] and turn_end >= segment['start']:
                            speakers.append(speaker)
                    except Exception as e:
                        app.logger.warning(f"Failed to process diarization turn: {str(e)}")
                        continue
                
                # Assign most common speaker
                if speakers:
                    from collections import Counter
                    segment['speaker'] = Counter(speakers).most_common(1)[0][0]
                else:
                    segment['speaker'] = 'SPEAKER_00'
            
            response = {
                'transcription': text,
                'language': result.get('language', 'en'),
                'segments': diarize_segments
            }
            
            # Generate embedding for the full transcription
            embedding = embedder.encode(text)
            
            # Store in Qdrant
            qdrant.upsert(
                collection_name='transcriptions',
                points=[
                    {
                        'id': int(datetime.datetime.now().timestamp() * 1000),
                        'vector': embedding.tolist(),
                        'payload': {
                            'text': text,
                            'language': response['language'],
                            'segments': response['segments'],
                            'filename': secure_filename(file.filename),
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                    }
                ]
            )
            
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
