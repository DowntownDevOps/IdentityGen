FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ninja-build \
    cmake \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    ffmpeg \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install moviepy first
RUN pip3 install --no-cache-dir moviepy==1.0.3

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Verify pydantic installation
RUN python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" && \
    python3 -c "import pydantic_core; print(f'Pydantic-core version: {pydantic_core.__version__}')"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src:/usr/local/lib/python3.10/site-packages:/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.10/site-packages
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Create necessary directories
RUN mkdir -p /app/src /app/models /app/storage

# Expose the port
EXPOSE 8000

# Run the application with uvicorn
CMD ["python3", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]