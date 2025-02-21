from dataclasses import dataclass
from typing import Optional
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
class ModelConfig:
    base_model_path: str = os.getenv("MODEL_PATH", "/app/models/checkpoints/sd_xl_base_1.0.safetensors")
    lora_path: Optional[str] = os.getenv("LORA_PATH")
    device: str = os.getenv("DEVICE", "cuda")
    dtype: str = os.getenv("DTYPE", "float16")
    storage: StorageConfig = StorageConfig()
    lora_training: LoRATrainingConfig = LoRATrainingConfig()
    
@dataclass
class GenerationConfig:
    num_inference_steps: int = 150  # Higher quality but safe number of steps
    guidance_scale: float = 7.5
    strength: float = 0.8
    seed: Optional[int] = None  # Random seed for reproducibility
    character_id: Optional[str] = None  # Used to identify which character LoRA to use
    
@dataclass
class APIConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "1")) 