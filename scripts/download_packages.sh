#!/bin/bash

# Create packages directory if it doesn't exist
mkdir -p packages

# Create a temporary container to download packages
docker run --rm \
    -v $(pwd)/packages:/packages \
    -v $(pwd)/requirements.txt:/requirements.txt \
    nvidia/cuda:11.8.0-runtime-ubuntu22.04 \
    bash -c "apt-get update && \
             apt-get install -y python3.10 python3-pip && \
             pip3 install --target=/packages -r /requirements.txt"

echo "Packages downloaded successfully to ./packages directory" 