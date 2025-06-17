# Use official Python 3.12 slim base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files to the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port Cloud Run will send requests to
EXPOSE $PORT

# Run your Streamlit app on port 8080
CMD ["streamlit", "run", "fifi-co-pilot.py", "--server.port=8080", "--server.address=0.0.0.0"]
