#!/bin/bash

# Function to check if command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        echo "❌ Error: $1"
        exit 1
    fi
}

echo "=== Character Generator Setup ==="

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models storage
check_status "Failed to create directories"

# Download Python packages for Docker
echo "📦 Downloading Python packages for Docker..."
bash scripts/download_packages.sh
check_status "Failed to download Python packages"

# Download models if needed
echo "📥 Downloading required models..."
bash scripts/download_models.sh
check_status "Failed to download models"

# Build and start Docker container
echo "🐳 Building and starting Docker container..."
docker compose up --build -d
check_status "Failed to start Docker container"

echo "✅ Setup completed successfully!"
echo "💡 The application should now be running in Docker"
echo "📝 Check docker compose logs for application status" 