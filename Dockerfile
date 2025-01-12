FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set GPU capabilities for RTX 4090
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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

# Copy the rest of the application
COPY . .

# Environment variables
ENV FLASK_ENV=production
ENV WHISPER_MODEL=large-v2

# Expose the application port
EXPOSE 2990

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:2990", "--timeout", "300", "--graceful-timeout", "300", "--keepalive", "75", "app:app"]
