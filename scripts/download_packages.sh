#!/bin/bash

# Create packages directory if it doesn't exist
mkdir -p packages

# Use CUDA 12.4 base image
docker run --rm \
    -v $(pwd)/packages:/packages \
    --gpus all \
    nvidia/cuda:12.4.1-devel-ubuntu22.04 \
    bash -c "
        # Install system dependencies
        apt-get update && \
        apt-get install -y \
        python3.10 \
        python3-pip \
        python3-dev \
        python3-venv \
        git \
        cmake \
        ninja-build \
        libcudnn9-cuda-12 \
        libcudnn9-dev-cuda-12 \
        && \
        
        # Create and activate virtual environment
        python3 -m venv /venv && \
        . /venv/bin/activate && \
        
        # Set CUDA environment variables
        export CUDACXX=/usr/local/cuda/bin/nvcc && \
        export PATH=/usr/local/cuda/bin:\$PATH && \
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
        
        # Add NVIDIA repository for CUDNN
        apt-get install -y wget && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update && \
        
        # Upgrade pip and install basics
        pip install --no-cache-dir -U pip && \
        pip install --no-cache-dir \
            packaging \
            setuptools \
            wheel || { echo 'Basic install failed'; exit 1; } && \
        
        # Install PyTorch and torchvision for CUDA 12.4 (cu124)
        pip install --no-cache-dir \
            torch==2.5.1+cu124 \
            torchvision==0.20.1+cu124 \
            --index-url https://download.pytorch.org/whl/cu124 || { echo 'PyTorch install failed'; exit 1; } && \
        
        # Install ML packages with compatible versions
        pip install --no-cache-dir xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 || { echo 'xformers install failed'; exit 1; } && \
        pip install --no-cache-dir triton==3.1.0 && \
        pip install --no-cache-dir accelerate==0.34.2 && \
        pip install --no-cache-dir transformers==4.44.2 && \
        pip install --no-cache-dir diffusers==0.30.3 && \
        pip install --no-cache-dir safetensors==0.4.5 && \
        pip install --no-cache-dir einops==0.8.0 && \
        
        # Install API and utility packages first
        pip install --no-cache-dir \
            fastapi==0.115.4 \
            uvicorn[standard]==0.32.0 \
            python-multipart==0.0.12 \
            tqdm==4.66.5 \
            requests==2.32.3 \
            peft==0.13.2 && \
        
        # Install OpenSora dependencies with updated version
        pip install --no-cache-dir \
            flash-attn==2.5.6 \
            rotary-embedding-torch==0.8.6 \
            einops-exts==0.0.4 \
            torch-fidelity==0.3.0 && \
            
        # Clone and install OpenSora
        git clone https://github.com/hpcaitech/Open-Sora.git && \
        cd Open-Sora && \
        pip install --no-cache-dir -e . && \
        cd .. && \
        
        # Install moviepy for video handling
        pip install --no-cache-dir moviepy==1.0.3 && \
        
        # Copy installed packages to mounted volume with better error handling
        rm -rf /packages/* && \
        mkdir -p /packages && \
        cp -r /venv/lib/python3.10/site-packages/* /packages/ && \
        pip freeze > /packages/requirements.txt && \
        echo 'Package copy completed successfully'
    " || { echo "Package download failed"; exit 1; }

echo "Packages downloaded successfully to ./packages directory"