from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class StorageConfig:
    base_characters_path: str = os.getenv("BASE_CHARACTERS_PATH", "/app/storage/base_characters")
    lora_models_path: str = os.getenv("LORA_MODELS_PATH", "/app/storage/lora_models")
    output_path: str = os.getenv("OUTPUT_PATH", "/app/storage/outputs")

@dataclass
class LoRATrainingConfig:
    num_train_epochs: int = 100
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    rank: int = 4
    mixed_precision: str = "fp16"

@dataclass
class VideoGenerationConfig:
    # Duration of the generated video in seconds
    duration: float = 5.0
    
    # Number of frames (must be 4n+1, e.g. 93, 77, 61, 45, 29)
    num_frames: int = 93
    
    # Frame rate of the output video
    fps: int = 24
    
    # Resolution must be multiple of 32
    width: int = 640
    height: int = 640
    
    # Number of inference steps
    num_inference_steps: int = 50
    
    # Guidance scale for keyframe generation
    guidance_scale: float = 8.5
    
    # Character consistency weight (0-1)
    # Higher values preserve character appearance better
    character_consistency: float = 0.8

@dataclass
class ModelConfig:
    base_model_path: str = os.getenv("MODEL_PATH", "/app/models/checkpoints/sd_xl_base_1.0.safetensors")
    lora_path: Optional[str] = os.getenv("LORA_PATH")
    device: str = os.getenv("DEVICE", "cuda")
    dtype: str = os.getenv("DTYPE", "float16")
    storage: StorageConfig = StorageConfig()
    lora_training: LoRATrainingConfig = LoRATrainingConfig()
    
    # Add paths for video models
    opensora_path: str = os.getenv("OPENSORA_PATH", "/app/models/opensora")
    
    # Add video config
    video: VideoGenerationConfig = VideoGenerationConfig()

@dataclass
class GenerationConfig:
    # Number of denoising steps - higher values give better quality but take longer
    num_inference_steps: int = 200
    
    # Controls how closely the image follows the prompt - higher values = more prompt adherence
    # but may reduce image quality. Values typically range from 7-9
    guidance_scale: float = 8.5
    
    # For img2img: how much to preserve of the original image (0-1)
    # 0 = completely new image, 1 = minimal changes to original
    strength: float = 0.75
    
    # Random seed for reproducible generations
    # Same seed + same parameters = same image
    seed: Optional[int] = 124124696969
    
    # ID of the character to use for generation
    # Used to load specific character LoRA or settings
    character_id: Optional[str] = None
    
    # Image dimensions - must be multiples of 8 for Stable Diffusion
    # Default is 1024x1024 for character portraits
    width: int = 1024
    height: int = 1024
    
    # Prompt enhancement controls
    # Whether to use the default prompt enhancements
    use_default_prompt_enhancements: bool = True
    
    # Custom prompt modifiers to use instead of the default ones
    # Only used if use_default_prompt_enhancements is False
    custom_prompt_modifiers: Optional[List[str]] = None
    
    # Whether to use the default negative prompt enhancements
    use_default_negative_enhancements: bool = True
    
    # Custom negative prompt modifiers to use instead of the default ones
    # Only used if use_default_negative_enhancements is False
    custom_negative_modifiers: Optional[List[str]] = None

@dataclass
class APIConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "1")) 