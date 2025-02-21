FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/models /app/storage

# Set Python path to include packages and app
ENV PYTHONPATH=/packages:/app:${PYTHONPATH}

# Add debug script
RUN echo '#!/bin/bash\necho "PYTHONPATH: $PYTHONPATH"\necho "Contents of /app:"\nls -la /app\necho "Contents of /app/src:"\nls -la /app/src\npython3 -c "import sys; print(sys.path)"\nexec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "src.main"] 