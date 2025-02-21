#!/bin/bash

# Set workspace path - using relative path for flexibility
WORKSPACE_PATH="./models"

# Set maximum number of concurrent downloads
MAX_CONCURRENT=3
current_downloads=0

# Add counters for total and completed downloads
total_downloads=0
completed_downloads=0
failed_downloads=0

# Function to check if command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        echo "❌ Error: $1"
        exit 1
    fi
}

# Function to download file
download_model() {
    local url="$1"
    local dest="$2"
    
    ((total_downloads++))
    
    local dir=$(dirname "$dest")
    if [ ! -d "$dir" ]; then
        echo "📁 Creating directory: $dir"
        mkdir -p "$dir"
    fi
    
    if [ -f "$dest" ]; then
        ((completed_downloads++))
        echo "✅ Skipping $(basename "$dest") - already exists ($completed_downloads/$total_downloads completed)"
        return
    fi
    
    while [ $current_downloads -ge $MAX_CONCURRENT ]; do
        sleep 1
        current_downloads=$(jobs -p | wc -l)
    done
    
    ((current_downloads++))
    (
        if wget --progress=bar:force:noscroll \
                --show-progress \
                -O "$dest.tmp" "$url" 2>&1 | \
            stdbuf -o0 awk '
            /[.] +[0-9][0-9]?[0-9]?%/ {
                printf "\r⏳ %-50s | %s | %s/s | %s | %d/%d downloads", 
                    substr(FILENAME, 1, 50),
                    $2, 
                    $3,
                    $1,
                    ENVIRON["completed_downloads"],
                    ENVIRON["total_downloads"]
            }
            ' FILENAME="$(basename "$dest")" completed_downloads="$completed_downloads" total_downloads="$total_downloads"; then
            mv "$dest.tmp" "$dest"
            ((completed_downloads++))
            echo -e "\n✅ Completed downloading $(basename "$dest") ($completed_downloads/$total_downloads completed)"
        else
            echo -e "\n❌ Failed downloading $(basename "$dest") ($completed_downloads/$total_downloads completed)"
            rm -f "$dest.tmp"
            ((failed_downloads++))
        fi
        ((current_downloads--))
    ) &
}

echo "📥 Starting model downloads..."
echo "🎯 Total files to download: $total_downloads"

# Base SDXL model
download_model \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    "$WORKSPACE_PATH/checkpoints/sd_xl_base_1.0.safetensors"

# LCM-SDXL model
download_model \
    "https://huggingface.co/latent-consistency/lcm-sdxl/resolve/main/diffusion_pytorch_model.safetensors" \
    "$WORKSPACE_PATH/checkpoints/lcm_sdxl.safetensors"

# CLIP models
download_model \
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin" \
    "$WORKSPACE_PATH/clip/clip-vit-large-patch14/pytorch_model.bin"

# ControlNet models
download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
    "$WORKSPACE_PATH/controlnet/control_v11p_sd15_openpose.pth"

# Wait for all background downloads to complete
wait

# Check if any downloads failed
if [ $failed_downloads -gt 0 ]; then
    echo "❌ $failed_downloads downloads failed. Please check the logs above and retry."
    exit 1
fi

echo "✅ All downloads completed successfully!" 