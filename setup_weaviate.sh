#!/bin/bash

# Define variables
DATA_DIR="./weaviate_data"
PYTHON_SETUP_SCRIPT="weaviate_setup.py"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Docker
if ! command_exists docker; then
    echo "Docker is not installed. Please install Docker before running this script."
    exit 1
fi

# Check for Docker Compose
if ! command_exists docker && ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose is not installed. Please install Docker Compose before running this script."
    exit 1
fi

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory at $DATA_DIR..."
    mkdir -p "$DATA_DIR"
fi

# Check if Weaviate service exists in docker-compose.yml
if ! grep -q "weaviate:" docker-compose.yml; then
    echo "Adding Weaviate service to docker-compose.yml..."
    # Add Weaviate service to docker-compose.yml
    cat >> docker-compose.yml <<EOL

  weaviate:
    image: semitechnologies/weaviate:latest
    container_name: weaviate
    ports:
      - "2500:8080"
    environment:
      - QUERY_DEFAULTS=top_k=10
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
    volumes:
      - weaviate-data:/var/lib/weaviate
    networks:
      - nca-network
    restart: unless-stopped
EOL

    # Add volume to volumes section if it doesn't exist
    if ! grep -q "weaviate-data:" docker-compose.yml; then
        sed -i '/^volumes:/a \  weaviate-data:' docker-compose.yml
    fi
fi

# Start Weaviate using the main docker-compose.yml
echo "Starting Weaviate..."
docker compose up -d weaviate

# Wait for Weaviate to be ready
echo "Waiting for Weaviate to be ready..."
until curl -s http://localhost:2500/v1/.well-known/ready | grep -q "OK"; do
    sleep 2
done

echo "Weaviate is ready."

# Optional: Run Python setup script if it exists
if [ -f "$PYTHON_SETUP_SCRIPT" ]; then
    if command_exists python3; then
        echo "Running Python setup script..."
        python3 "$PYTHON_SETUP_SCRIPT"
    else
        echo "Python3 is not installed. Skipping Python setup."
    fi
fi
