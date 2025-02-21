#!/bin/bash

# Set workspace path - this will be on the host machine
WORKSPACE_PATH="./models"

# Add counters for total and completed downloads
total_downloads=0
completed_downloads=0

# Define all downloads upfront
declare -a downloads=(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors|checkpoints/sd_xl_base_1.0.safetensors"
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin|clip/clip-vit-large-patch14/pytorch_model.bin"
)

# Calculate total downloads
total_downloads=${#downloads[@]}

# Function to download file
download_model() {
    local url="$1"
    local dest="$2"
    
    local dir=$(dirname "$dest")
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
    
    if [ -f "$dest" ]; then
        ((completed_downloads++))
        echo "✅ Skipping $(basename "$dest") - already exists ($completed_downloads/$total_downloads completed)"
        return
    fi
    
    if wget --progress=bar:force:noscroll \
            --show-progress \
            -O "$dest" "$url" 2>&1 | \
        stdbuf -o0 awk -v file="$(basename "$dest")" \
                    -v completed="$completed_downloads" \
                    -v total="$total_downloads" '
            BEGIN { 
                file_total="?B"
                file_current="0B"
                file_percent="0%"
                file_speed="0B/s"
            }
            /[.] +[0-9][0-9]?[0-9]?%/ { 
                file_percent=$2
                file_speed=$3
                if ($1 ~ /[GMKB]/) { 
                    file_current=$1
                    if ($4 ~ /[GMKB]/) file_total=$4
                }
                printf "\r⏳ %-40s | %6s of %6s | %6s | %9s/s | %d/%d files", 
                    file,
                    file_current,
                    file_total,
                    file_percent,
                    file_speed,
                    completed,
                    total
            }
            ' FILENAME="$(basename "$dest")" completed_downloads="$completed_downloads" total_downloads="$total_downloads"; then
        ((completed_downloads++))
        echo -e "\n✅ Completed downloading $(basename "$dest")"
    else
        echo -e "\n❌ Failed downloading $(basename "$dest") ($completed_downloads/$total_downloads completed)"
        rm -f "$dest"
        exit 1
    fi
}

echo "📥 Starting model downloads..."
echo "🎯 Total files to download: $total_downloads"
echo

# Process all downloads
for download in "${downloads[@]}"; do
    IFS="|" read -r url path <<< "$download"
    download_model "$url" "$WORKSPACE_PATH/$path"
done

# Check if any downloads failed
if [ $? -ne 0 ]; then
    echo "❌ Some downloads failed. Please check the logs above."
    exit 1
fi

echo "✅ All downloads completed successfully!" 