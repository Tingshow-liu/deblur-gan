# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    patch \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app /app

# Copy the model directory
COPY model /app/model

# Create the output directory
RUN mkdir -p /app/output

# Expose the port for FastAPI
EXPOSE 8000

# Environment variables for the application
ENV MODEL_PATH="/app/model/generator_epoch_220.pth"
ENV OUTPUT_DIR="/app/output"
ENV INPUT_NC=3
ENV OUTPUT_NC=3
ENV NGF=64
ENV N_BLOCKS=6

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "deblurgan_inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
