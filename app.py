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
import warnings

# Suppress FutureWarning for _register_pytree_node
warnings.filterwarnings('ignore', category=FutureWarning, 
                       module='transformers.utils.generic')

# Configure PyTorch settings
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    # Enable TF32 explicitly as recommended
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Initialize Qdrant client
qdrant = QdrantClient(os.getenv('QDRANT_URL', 'http://localhost:6333'))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250MB

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

# Configuration
MODEL_NAME = os.getenv('WHISPER_MODEL', 'large-v2')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
MAX_FILE_SIZE = 250 * 1024 * 1024  # 250MB

# Verify GPU availability
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

def format_segment_text(segment: dict) -> str:
    """Format a single segment into a readable string."""
    text = segment['text'].strip()
    if not text:
        return ""
    return f"(Speaker {segment['speaker']}) {text}"

def merge_sequential_speaker_segments(segments: list) -> list:
    """Merge consecutive segments from the same speaker into single segments."""
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for segment in segments[1:]:
        if (segment['speaker'] == current['speaker'] and 
            segment['start'] - current['end'] < 1.0):  # 1 second threshold
            # Merge the segments
            current['end'] = segment['end']
            current['text'] = f"{current['text'].strip()} {segment['text'].strip()}"
        else:
            merged.append(current)
            current = segment.copy()
    
    merged.append(current)
    return merged

def format_response(result: dict) -> dict:
    """Format the transcription result into a readable response."""
    # Merge sequential segments from the same speaker
    processed_segments = merge_sequential_speaker_segments(result['segments'])
    
    # Format each segment
    formatted_segments = [format_segment_text(segment) for segment in processed_segments]
    
    # Join with newlines
    transcription = '\n'.join(formatted_segments)
    
    return {
        'transcription': transcription,
        'language': result.get('language', 'en'),
        'segments': processed_segments
    }

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
        filename = secure_filename(file.filename)
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        app.logger.info(f"Starting enhanced transcription for file: {filename}")
        
        # Step 1: Initial transcription with larger batch size
        result = model.transcribe(temp_path, batch_size=32)
        app.logger.info(f"Initial transcription complete. Language: {result.get('language', 'unknown')}")

        # Step 2: Enhanced alignment with word-level timestamps
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
            app.logger.info("Alignment completed with word-level timestamps")
        except ValueError as ve:
            app.logger.error(f"Hugging Face token error: {str(ve)}")
            raise
        except Exception as e:
            app.logger.error(f"Diarization failed: {str(e)}")
            app.logger.exception("Detailed diarization error:")
            pass

        # Step 3: Enhanced diarization
        try:
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable not set")
            
            # Verify token format
            if not hf_token.startswith('hf_'):
                raise ValueError("Invalid HF_TOKEN format. Token should start with 'hf_'")

            # Initialize diarization pipeline with supported parameters
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device,
                cache_dir="/tmp/huggingface_cache"
            )

            # Run diarization with supported parameters
            diarize_segments = diarize_model(
                temp_path,
                min_speakers=int(os.getenv('DIARIZATION_MIN_SPEAKERS', 2)),
                max_speakers=int(os.getenv('DIARIZATION_MAX_SPEAKERS', 5))
            )
            
            # Step 4: Improved speaker assignment
            result = whisperx.assign_word_speakers(
                diarize_segments, 
                result
            )

            # Post-process speaker assignments
            speaker_segments = {}
            for segment in result['segments']:
                if 'speaker' not in segment:
                    continue
                    
                speaker = segment['speaker']
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)

            app.logger.info(f"Diarization completed successfully. Found {len(speaker_segments)} speakers")
            
        except Exception as e:
            app.logger.error(f"Diarization failed: {str(e)}")
            app.logger.exception("Detailed diarization error:")
            # Fallback to basic speaker separation
            for i, segment in enumerate(result['segments']):
                if i > 0:
                    prev_segment = result['segments'][i-1]
                    if segment['start'] - prev_segment['end'] > 1.0:
                        segment['speaker'] = f'SPEAKER_{(int(prev_segment["speaker"].split("_")[1]) + 1) % 3:02d}'
                    else:
                        segment['speaker'] = prev_segment['speaker']
                else:
                    segment['speaker'] = 'SPEAKER_00'

        # Format and create response
        response = format_response(result)

        # Store in Qdrant
        try:
            embedding = embedder.encode(response['transcription'])
            qdrant.upsert(
                collection_name='transcriptions',
                points=[{
                    'id': int(datetime.datetime.now().timestamp() * 1000),
                    'vector': embedding.tolist(),
                    'payload': {
                        'text': response['transcription'],
                        'language': response['language'],
                        'segments': response['segments'],
                        'filename': filename,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                }]
            )
            app.logger.info("Successfully stored in Qdrant")
        except Exception as e:
            app.logger.error(f"Qdrant storage failed: {str(e)}")

        # Clean up
        os.remove(temp_path)
        
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 2990))
    app.run(host='0.0.0.0', port=port)