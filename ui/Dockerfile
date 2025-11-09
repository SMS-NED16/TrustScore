# TrustScore UI Dockerfile for RunPod
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt /app/root_requirements.txt
COPY ui/requirements.txt /app/ui_requirements.txt

# Install root dependencies first
RUN pip install --no-cache-dir -r /app/root_requirements.txt

# Install UI dependencies
RUN pip install --no-cache-dir -r /app/ui_requirements.txt

# Copy the entire project
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port (RunPod will map this)
EXPOSE 5000

# Set working directory to ui folder
WORKDIR /app/ui

# Run the Flask app
CMD ["python", "app.py"]

