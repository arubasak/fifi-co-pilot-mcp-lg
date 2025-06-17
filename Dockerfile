# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code from your host to your container at /app
# This assumes your Dockerfile is in the root of your project directory
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the health check to ensure the container is running correctly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD [ "streamlit", "healthcheck" ]

# Define the command to run your app
# The --server.port and --server.address options are added for compatibility with cloud hosting
CMD ["streamlit", "run", "fifi-co-pilot.py", "--server.port=8501", "--server.address=0.0.0.0"]