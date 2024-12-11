# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the application code
COPY app /app

# Expose Streamlit default port
EXPOSE 8501

# Environment variable for Streamlit to run seamlessly inside Docker
ENV STREAMLIT_SERVER_ENABLE_CORS="false"
ENV STREAMLIT_SERVER_PORT="8501"
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
