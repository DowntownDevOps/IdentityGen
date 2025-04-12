#!/bin/bash

echo "=== Character Generator Setup ==="

# Load environment variables from .env file
if [ -f .env ]; then
    echo "📝 Loading environment variables from .env..."
    set -a  # Automatically export all variables
    source .env
    set +a
else
    echo "⚠️ No .env file found, proceeding without it..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models/{checkpoints,clip}
mkdir -p storage/{base_characters,lora_models,outputs,db}
mkdir -p packages

# Download models if they don't exist
if [ ! -f "models/checkpoints/sd_xl_base_1.0.safetensors" ]; then
    echo "📥 Downloading required models..."
    python3 scripts/download_models_prereq.py
fi

# Check if Llama 3 model exists in Ollama
echo "🔍 Checking for Llama 3 model in Ollama..."
if ! docker exec ollama ollama list | grep -q "llama3"; then
    echo "📥 Pulling Llama 3 model for Ollama..."
    docker exec ollama ollama pull llama3
else
    echo "✅ Llama 3 model already exists in Ollama."
fi

# Download packages if they don't exist
if [ ! -d "packages/torch" ]; then
    echo "📥 Downloading required packages..."
    bash scripts/download_packages.sh
fi

# Build and start Docker container
echo "🐳 Building and starting Docker container..."
docker compose up --build -d

echo "✅ Setup complete!"