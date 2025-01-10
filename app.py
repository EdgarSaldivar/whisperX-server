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
                result = model.transcribe(temp_path, batch_size=16)
                
                # Align output
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
                
                # Get speaker count configuration
                min_speakers = int(os.getenv('DIARIZATION_MIN_SPEAKERS', 1))
                max_speakers = int(os.getenv('DIARIZATION_MAX_SPEAKERS', 5))
                
                app.logger.info(f"Configuring diarization with {min_speakers}-{max_speakers} speakers")
                
                # Diarize with speaker count estimation
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                diarize_segments = diarize_model(
                    temp_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                
                # Assign speakers with detailed logging
                app.logger.info("Assigning speakers to words...")
                try:
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # Ensure all segments have a speaker
                    for segment in result['segments']:
                        if 'speaker' not in segment:
                            segment['speaker'] = 'UNKNOWN'
                            app.logger.warning(f"Added default speaker to segment: {segment}")
                    
                    # Log speaker distribution
                    speaker_counts = {}
                    for segment in result['segments']:
                        speaker = segment.get('speaker', 'UNKNOWN')
                        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                    app.logger.info(f"Speaker distribution: {speaker_counts}")
                except Exception as e:
                    app.logger.error(f"Speaker assignment failed: {str(e)}")
                    # Fallback to single speaker
                    for segment in result['segments']:
                        segment['speaker'] = 'SPEAKER_00'
                    app.logger.info("Falling back to single speaker mode")
                
                app.logger.info("Diarization completed successfully")
                
                # Clean up models
                del align_model
                del diarize_model
                import gc
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Use the WhisperX assigned speaker segments
                diarize_segments = result['segments']
            except Exception as e:
                app.logger.error(f"Diarization failed: {str(e)}")
                diarize_segments = result['segments']
                for segment in diarize_segments:
                    segment['speaker'] = 'SPEAKER_00'
            
            # Combine text with speaker information
            try:
                text = ' '.join(
                    f"[Speaker {segment.get('speaker', 'UNKNOWN')}] {segment['text'].strip()}"
                    for segment in result['segments']
                )
            except KeyError as e:
                app.logger.error(f"Missing required field in segment: {str(e)}")
                app.logger.error(f"Problematic segment: {segment}")
                raise ValueError(f"Invalid segment format: missing {str(e)}")
            
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
            
            # Simplified response with only final transcription
            response = {
                'transcription': text,
                'language': result.get('language', 'en')
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
