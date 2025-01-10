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
        filename = secure_filename(file.filename)
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        app.logger.info(f"Starting enhanced transcription for file: {filename}")
        
        # Step 1: Initial transcription with larger batch size for better context
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
        except Exception as e:
            app.logger.error(f"Alignment failed: {str(e)}")
            pass

        # Step 3: Enhanced diarization with improved parameters
        try:
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN not found")

            # Initialize diarization pipeline with enhanced settings
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device,
                sampling_rate=16000,  # Explicitly set sampling rate
                pipeline_config={
                    "vad": {
                        "min_speech_duration_ms": 250,  # Reduced from default for better separation
                        "min_silence_duration_ms": 100,  # Reduced to detect quick speaker changes
                        "speech_pad_ms": 30  # Added small padding
                    },
                    "speaker_detection": {
                        "min_speakers": int(os.getenv('DIARIZATION_MIN_SPEAKERS', 2)),  # Default to expecting at least 2 speakers
                        "max_speakers": int(os.getenv('DIARIZATION_MAX_SPEAKERS', 5)),
                        "audio_duration_threshold": 0.5,  # Minimum duration for speaker segment
                        "speech_activity_threshold": 0.4  # Lower threshold for speech detection
                    }
                }
            )

            # Run diarization with enhanced parameters
            diarize_segments = diarize_model(
                temp_path,
                min_speakers=int(os.getenv('DIARIZATION_MIN_SPEAKERS', 2)),
                max_speakers=int(os.getenv('DIARIZATION_MAX_SPEAKERS', 5)),
                silence_threshold=0.4,  # More sensitive silence detection
                segment_length=10,  # Shorter segments for better speaker detection
                step_size=3  # Smaller step size for more precise segmentation
            )
            
            # Step 4: Improved speaker assignment
            result = whisperx.assign_word_speakers(
                diarize_segments, 
                result,
                speaker_embeddings=True,  # Enable speaker embeddings
                min_speaker_duration=0.5  # Minimum duration for speaker segments
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

            # Merge adjacent segments from same speaker
            processed_segments = []
            current_segment = None
            
            for segment in result['segments']:
                if current_segment is None:
                    current_segment = segment.copy()
                    continue
                    
                if (segment['speaker'] == current_segment['speaker'] and 
                    segment['start'] - current_segment['end'] < 0.5):  # 500ms threshold
                    # Merge segments
                    current_segment['end'] = segment['end']
                    current_segment['text'] += ' ' + segment['text']
                else:
                    processed_segments.append(current_segment)
                    current_segment = segment.copy()
            
            if current_segment:
                processed_segments.append(current_segment)

            # Update result with processed segments
            result['segments'] = processed_segments

            app.logger.info(f"Diarization completed. Detected {len(speaker_segments)} speakers")
            
        except Exception as e:
            app.logger.error(f"Diarization failed: {str(e)}")
            # Fallback to basic speaker separation
            for i, segment in enumerate(result['segments']):
                # Simple heuristic: assign different speakers based on time gaps
                if i > 0:
                    prev_segment = result['segments'][i-1]
                    if segment['start'] - prev_segment['end'] > 1.0:  # 1 second gap
                        segment['speaker'] = f'SPEAKER_{(int(prev_segment["speaker"].split("_")[1]) + 1) % 3:02d}'
                    else:
                        segment['speaker'] = prev_segment['speaker']
                else:
                    segment['speaker'] = 'SPEAKER_00'

        # Create response with processed segments
        text = ' '.join(
            f"[Speaker {segment['speaker']}] {segment['text'].strip()}"
            for segment in result['segments']
        )

        response = {
            'transcription': text,
            'language': result.get('language', 'en'),
            'segments': result['segments']
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
