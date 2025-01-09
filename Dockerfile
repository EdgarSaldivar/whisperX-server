# Use GPU-enabled Python image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install gunicorn for production
RUN python3 -m pip install gunicorn python-dotenv


# Create transcription storage directory
RUN mkdir -p /transcriptions

# Copy the rest of the application
COPY . .

# Environment variables
ENV FLASK_ENV=production
ENV TRANSCRIPTION_DIR=/transcriptions
ENV WHISPER_MODEL=large-v2

# Expose the application port
EXPOSE 2990

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:2990", "app:app"]
