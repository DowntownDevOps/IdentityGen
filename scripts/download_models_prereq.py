#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
from typing import List, Tuple

# Define downloads: (url, local_path)
DOWNLOADS = [
    (
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "checkpoints/sd_xl_base_1.0.safetensors"
    ),
    (
        "https://huggingface.co/latent-consistency/lcm-sdxl/resolve/main/diffusion_pytorch_model.safetensors",
        "checkpoints/lcm_sdxl.safetensors"
    ),
    (
        "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin",
        "clip/clip-vit-large-patch14/pytorch_model.bin"
    ),
]

def get_huggingface_token():
    """Get token from environment or prompt user"""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("\n⚠️  Hugging Face token not found in environment")
        print("Please enter your Hugging Face token (from https://huggingface.co/settings/tokens):")
        token = input("> ").strip()
        if not token:
            print("❌ No token provided")
            sys.exit(1)
    return token

def download_file(url: str, dest_path: Path, desc: str = None, token: str = None) -> bool:
    """Download a file with progress bar"""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        desc = desc or dest_path.name
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
            with open(dest_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {str(e)}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    models_dir = Path("./models")
    print("=== Character Generator Model Download ===")
    
    token = get_huggingface_token()
    failed_downloads = []
    
    for url, rel_path in DOWNLOADS:
        dest_path = models_dir / rel_path
        if dest_path.exists():
            print(f"✅ Skipping {dest_path.name} - already exists")
            continue
        
        print(f"\nDownloading {dest_path.name}...")
        if not download_file(url, dest_path, token=token):
            failed_downloads.append(dest_path.name)
    
    if failed_downloads:
        print("\n❌ The following downloads failed:")
        for name in failed_downloads:
            print(f"  - {name}")
        sys.exit(1)
    
    print("\n✅ All downloads completed successfully!")

if __name__ == "__main__":
    main() 