import os
import shutil
import json
from pathlib import Path

def setup_model_directory():
    # Create base model directory
    model_dir = Path("models/sdxl")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model_index.json
    model_index = {
        "_class_name": "StableDiffusionXLPipeline",
        "_diffusers_version": "0.21.4",
        "feature_extractor": ["CLIPImageProcessor", "CLIPImageProcessor"],
        "safety_checker": None,
        "scheduler": ["DDIMScheduler", "DDIMScheduler"],
        "text_encoder": ["CLIPTextModel", "CLIPTextModel"],
        "text_encoder_2": ["CLIPTextModel", "CLIPTextModel"],
        "tokenizer": ["CLIPTokenizer", "CLIPTokenizer"],
        "tokenizer_2": ["CLIPTokenizer", "CLIPTokenizer"],
        "unet": ["UNet2DConditionModel"],
        "vae": ["AutoencoderKL"]
    }
    
    with open(model_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    
    # Copy the safetensors file to the model directory
    source_file = Path("models/checkpoints/sd_xl_base_1.0.safetensors")
    target_file = model_dir / "model.safetensors"
    
    if not target_file.exists():
        print(f"Copying {source_file} to {target_file}")
        shutil.copy2(source_file, target_file)
    
    # Create required subdirectories
    (model_dir / "text_encoder").mkdir(exist_ok=True)
    (model_dir / "text_encoder_2").mkdir(exist_ok=True)
    (model_dir / "tokenizer").mkdir(exist_ok=True)
    (model_dir / "tokenizer_2").mkdir(exist_ok=True)
    (model_dir / "unet").mkdir(exist_ok=True)
    (model_dir / "vae").mkdir(exist_ok=True)
    
    print("Model directory setup complete!")
    print(f"Model path: {model_dir.absolute()}")
    print("Note: You may need to download additional model components from Hugging Face")

if __name__ == "__main__":
    setup_model_directory() 