import os
import uuid
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import logging
from typing import Optional, Dict, List
from config import ModelConfig, StorageConfig, LoRATrainingConfig, GenerationConfig
import datetime
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CharacterDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class CharacterState:
    INITIAL = "initial"  # Just created, no approved base image
    BASE_APPROVED = "base_approved"  # Has approved base image
    GENERATING_TRAINING = "generating_training"  # Generating training data
    TRAINING = "training"  # LoRA training in progress
    READY = "ready"  # LoRA trained and ready for use
    ERROR = "error"  # Something went wrong

DEFAULT_TRAINING_PROMPTS = [
    "full body shot, standing in a neutral pose, white background, front view",
    "detailed portrait shot of face and shoulders, white background, front view",
    "full body shot, standing in a neutral pose, white background, side view",
    "full body shot, standing in a neutral pose, white background, 3/4 view",
    "full body shot, standing in a neutral pose, white background, back view"
]

class CharacterManager:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.storage_config = model_config.storage
        self.lora_config = model_config.lora_training
        
        # Ensure storage directories exist
        os.makedirs(self.storage_config.base_characters_path, exist_ok=True)
        os.makedirs(self.storage_config.lora_models_path, exist_ok=True)
        os.makedirs(self.storage_config.output_path, exist_ok=True)

    def create_initial_character(self, prompt: str, existing_image: Optional[Image.Image] = None) -> Dict:
        """Create a new character either from a description or existing image"""
        character_id = str(uuid.uuid4())
        character_dir = Path(self.storage_config.base_characters_path) / character_id
        os.makedirs(character_dir, exist_ok=True)
        
        # Create character metadata
        metadata = {
            "id": character_id,
            "prompt": prompt,
            "state": CharacterState.INITIAL,
            "base_image_approved": False,
            "training_images": [],
            "creation_timestamp": str(datetime.datetime.now()),
            "last_modified": str(datetime.datetime.now())
        }
        
        if existing_image:
            # Save uploaded image as base
            image_path = character_dir / "base.png"
            existing_image.save(image_path)
            metadata["base_image_path"] = str(image_path)
            metadata["base_image_source"] = "uploaded"
        else:
            metadata["base_image_source"] = "generated"
            
        # Save metadata
        self._save_metadata(character_id, metadata)
        
        return metadata

    def save_generated_base(self, character_id: str, image: Image.Image) -> Dict:
        """Save a generated image as a potential base image"""
        character_dir = Path(self.storage_config.base_characters_path) / character_id
        if not character_dir.exists():
            raise ValueError(f"Character {character_id} not found")
            
        # Save the generated image
        image_path = character_dir / "base_generated.png"
        image.save(image_path)
        
        # Update metadata
        metadata = self._load_metadata(character_id)
        metadata["latest_generated_base"] = str(image_path)
        metadata["last_modified"] = str(datetime.datetime.now())
        self._save_metadata(character_id, metadata)
        
        return metadata

    def approve_base_image(self, character_id: str, use_latest: bool = True) -> Dict:
        """Approve a base image for a character"""
        metadata = self._load_metadata(character_id)
        
        if use_latest and "latest_generated_base" in metadata:
            # Move latest generated to be the base
            latest_path = Path(metadata["latest_generated_base"])
            base_path = latest_path.parent / "base.png"
            
            if latest_path.exists():
                # If there's an existing base, remove it
                if base_path.exists():
                    base_path.unlink()
                    
                # Move latest to be the base
                latest_path.rename(base_path)
                metadata["base_image_path"] = str(base_path)
                metadata.pop("latest_generated_base", None)
        
        # Update approval status
        metadata["base_image_approved"] = True
        metadata["state"] = CharacterState.BASE_APPROVED
        metadata["last_modified"] = str(datetime.datetime.now())
        metadata.pop("error", None)  # Clear any previous errors
        
        self._save_metadata(character_id, metadata)
        
        return {
            "status": "approved",
            "character": metadata,
            "message": "Base image approved. Ready for training data generation."
        }

    def get_character_info(self, character_id: str) -> Dict:
        """Get current character information and state"""
        return self._load_metadata(character_id)

    def _save_metadata(self, character_id: str, metadata: Dict):
        """Save character metadata to disk"""
        character_dir = Path(self.storage_config.base_characters_path) / character_id
        metadata_path = character_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, character_id: str) -> Dict:
        """Load character metadata from disk"""
        character_dir = Path(self.storage_config.base_characters_path) / character_id
        metadata_path = character_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Character {character_id} not found")
            
        with open(metadata_path) as f:
            return json.load(f)

    def get_character_lora_path(self, character_id: str) -> str:
        """Get the path to a character's LoRA weights"""
        return str(Path(self.storage_config.lora_models_path) / character_id / "lora.safetensors")

    def start_training(self, character_id: str, model_handler) -> Dict:
        """Start LoRA training for a character"""
        metadata = self._load_metadata(character_id)
        
        if len(metadata.get("training_images", [])) < 5:
            raise ValueError("Need at least 5 training images to start training")
            
        # Update state
        metadata["state"] = CharacterState.TRAINING
        metadata["training_start"] = str(datetime.datetime.now())
        metadata["training_progress"] = 0
        self._save_metadata(character_id, metadata)
        
        try:
            # Start training in background
            self._train_character_lora(character_id, model_handler)
            return metadata
        except Exception as e:
            metadata["state"] = CharacterState.ERROR
            metadata["error"] = str(e)
            self._save_metadata(character_id, metadata)
            raise

    def get_training_status(self, character_id: str) -> Dict:
        """Get current training status"""
        metadata = self._load_metadata(character_id)
        return {
            "state": metadata.get("state"),
            "progress": metadata.get("training_progress", 0),
            "error": metadata.get("error"),
            "training_start": metadata.get("training_start"),
            "last_update": metadata.get("last_modified")
        }

    def _train_character_lora(self, character_id: str, model_handler):
        """Train a LoRA model for the character"""
        try:
            metadata = self._load_metadata(character_id)
            character_dir = Path(self.storage_config.base_characters_path) / character_id
            
            # Prepare dataset
            dataset = CharacterDataset(metadata["training_images"])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Initialize model for training
            logger.info("Preparing model for training...")
            torch.cuda.empty_cache()  # Clear GPU memory
            
            unet = model_handler.pipeline.unet
            unet = prepare_model_for_kbit_training(unet)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.lora_config.rank,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                bias="none",
                alpha=self.lora_config.alpha
            )
            
            unet = get_peft_model(unet, lora_config)
            unet.train()  # Set to training mode
            
            # Training configuration
            optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
            num_epochs = self.lora_config.num_epochs
            
            # Save initial checkpoint
            checkpoint_dir = character_dir / "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._save_training_checkpoint(checkpoint_dir / "initial.pt", {
                "model_state": unet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": 0,
                "batch": 0
            })
            
            # Training loop
            total_steps = len(dataloader) * num_epochs
            current_step = 0
            running_loss = 0.0
            
            logger.info(f"Starting training for {num_epochs} epochs...")
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        # Move batch to device
                        batch = batch.to(model_handler.device)
                        
                        # Forward pass
                        with torch.cuda.amp.autocast():  # Mixed precision
                            loss = unet(batch).loss
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)  # Gradient clipping
                        
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Update metrics
                        current_step += 1
                        running_loss += loss.item()
                        epoch_loss += loss.item()
                        
                        # Update progress every few steps
                        if current_step % 5 == 0:
                            progress = (current_step / total_steps) * 100
                            avg_loss = running_loss / 5
                            metadata["training_progress"] = progress
                            metadata["training_loss"] = avg_loss
                            metadata["last_modified"] = str(datetime.datetime.now())
                            self._save_metadata(character_id, metadata)
                            running_loss = 0.0
                        
                        # Save checkpoint every 100 steps
                        if current_step % 100 == 0:
                            self._save_training_checkpoint(checkpoint_dir / f"step_{current_step}.pt", {
                                "model_state": unet.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "epoch": epoch,
                                "batch": batch_idx,
                                "loss": avg_loss
                            })
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            # Handle OOM error
                            torch.cuda.empty_cache()
                            logger.warning(f"OOM error at step {current_step}, attempting to recover...")
                            continue
                        raise
                
                # Log epoch metrics
                avg_epoch_loss = epoch_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
                metadata["current_epoch"] = epoch + 1
                metadata["epoch_loss"] = avg_epoch_loss
                self._save_metadata(character_id, metadata)
            
            # Save final model
            logger.info("Training completed, saving model...")
            output_dir = Path(self.storage_config.lora_models_path) / character_id
            os.makedirs(output_dir, exist_ok=True)
            unet.save_pretrained(output_dir)
            
            # Cleanup
            torch.cuda.empty_cache()
            
            # Update metadata
            metadata["state"] = CharacterState.READY
            metadata["lora_path"] = str(output_dir / "adapter_model.bin")
            metadata["training_completed"] = str(datetime.datetime.now())
            metadata["final_loss"] = avg_epoch_loss
            metadata["last_modified"] = str(datetime.datetime.now())
            self._save_metadata(character_id, metadata)
            
            logger.info(f"Training completed successfully for character {character_id}")
            
        except Exception as e:
            logger.error(f"Training failed for character {character_id}: {str(e)}")
            metadata = self._load_metadata(character_id)
            metadata["state"] = CharacterState.ERROR
            metadata["error"] = str(e)
            metadata["last_modified"] = str(datetime.datetime.now())
            self._save_metadata(character_id, metadata)
            raise

    def _save_training_checkpoint(self, path: Path, state: Dict):
        """Save a training checkpoint"""
        try:
            torch.save(state, path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

    def resume_training(self, character_id: str, model_handler, checkpoint_path: Optional[str] = None) -> Dict:
        """Resume training from the latest checkpoint"""
        metadata = self._load_metadata(character_id)
        if metadata["state"] not in [CharacterState.TRAINING, CharacterState.ERROR]:
            raise ValueError("Character must be in training or error state to resume")
            
        # Find latest checkpoint if not specified
        if not checkpoint_path:
            checkpoint_dir = Path(self.storage_config.base_characters_path) / character_id / "checkpoints"
            checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
            if not checkpoints:
                raise ValueError("No checkpoints found to resume from")
            checkpoint_path = str(checkpoints[-1])
            
        # Update state and start training
        metadata["state"] = CharacterState.TRAINING
        metadata["training_resumed"] = str(datetime.datetime.now())
        metadata["resume_checkpoint"] = checkpoint_path
        self._save_metadata(character_id, metadata)
        
        try:
            self._train_character_lora(character_id, model_handler, checkpoint_path)
            return metadata
        except Exception as e:
            metadata["state"] = CharacterState.ERROR
            metadata["error"] = str(e)
            self._save_metadata(character_id, metadata)
            raise

    def generate_training_data(
        self,
        character_id: str,
        num_variations: int = 1,
        custom_prompts: Optional[List[str]] = None,
        model_handler = None
    ) -> Dict:
        """Generate training data variations using the base image"""
        metadata = self._load_metadata(character_id)
        
        if metadata["state"] != CharacterState.BASE_APPROVED:
            raise ValueError("Character must have an approved base image before generating training data")
            
        if not model_handler:
            raise ValueError("Model handler is required for generating training data")
            
        character_dir = Path(self.storage_config.base_characters_path) / character_id
        training_dir = character_dir / "training_data"
        os.makedirs(training_dir, exist_ok=True)
        
        # Update state
        metadata["state"] = CharacterState.GENERATING_TRAINING
        metadata["training_images"] = []
        metadata["training_prompts"] = custom_prompts if custom_prompts else DEFAULT_TRAINING_PROMPTS
        self._save_metadata(character_id, metadata)
        
        base_image = Image.open(metadata["base_image_path"])
        
        # Extract core character description (first sentence only)
        base_prompt = metadata["prompt"].split('.')[0] + "."
        
        try:
            # Generate variations
            prompts = custom_prompts if custom_prompts else DEFAULT_TRAINING_PROMPTS
            if len(prompts) < num_variations:
                # Repeat prompts if we need more variations
                prompts = prompts * (num_variations // len(prompts) + 1)
            prompts = prompts[:num_variations]
            
            # Encode base image to latents
            reference_latents = model_handler.encode_reference_image(metadata["base_image_path"])
            
            # Generate a list of random seeds
            seeds = torch.randint(0, 2**32 - 1, (num_variations,)).tolist()
            
            for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
                # Combine core character description with training prompt
                full_prompt = f"{base_prompt} {prompt}"
                
                # Create generation config with optimized parameters
                gen_config = GenerationConfig(
                    strength=0.7,  # Balance between keeping character features and allowing variation
                    guidance_scale=7.5,  # Standard guidance scale
                    num_inference_steps=150,  # Higher quality but safe number of steps
                    seed=seed  # Use unique seed for each variation
                )
                
                # Gerate variation using base pipeline
                variation = model_handler.generate_image(
                    prompt=full_prompt,
                    reference_latents=reference_latents,
                    gen_config=gen_config
                )
                
                # Save variation
                variation_path = training_dir / f"variation_{i+1}.png"
                variation.save(variation_path)
                metadata["training_images"].append(str(variation_path))
                
                # Update metadata after each successful generation
                metadata["last_modified"] = str(datetime.datetime.now())
                self._save_metadata(character_id, metadata)
                
            # Update state after successful generation
            metadata["state"] = CharacterState.BASE_APPROVED  # Ready for training
            self._save_metadata(character_id, metadata)
            
            return metadata
            
        except Exception as e:
            metadata["state"] = CharacterState.ERROR
            metadata["error"] = str(e)
            self._save_metadata(character_id, metadata)
            raise

    def get_training_image(self, character_id: str, image_index: int) -> Optional[str]:
        """Get the path to a specific training image"""
        metadata = self._load_metadata(character_id)
        if not metadata.get("training_images"):
            return None
        try:
            return metadata["training_images"][image_index]
        except IndexError:
            return None

    def remove_training_image(self, character_id: str, image_index: int) -> Dict:
        """Remove a training image from the set"""
        metadata = self._load_metadata(character_id)
        if not metadata.get("training_images"):
            raise ValueError("No training images exist")
            
        try:
            image_path = Path(metadata["training_images"][image_index])
            if image_path.exists():
                image_path.unlink()
            metadata["training_images"].pop(image_index)
            metadata["last_modified"] = str(datetime.datetime.now())
            self._save_metadata(character_id, metadata)
            return metadata
        except IndexError:
            raise ValueError(f"Image index {image_index} not found")

    def regenerate_training_image(
        self,
        character_id: str,
        image_index: int,
        model_handler,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """Regenerate a specific training image"""
        metadata = self._load_metadata(character_id)
        if not metadata.get("training_images"):
            raise ValueError("No training images exist")
            
        try:
            base_image = Image.open(metadata["base_image_path"])
            base_prompt = metadata["prompt"]
            
            # Get or create prompt
            if custom_prompt:
                full_prompt = f"{base_prompt}, {custom_prompt}"
            else:
                variation_prompt = metadata.get("training_prompts", DEFAULT_TRAINING_PROMPTS)[image_index]
                full_prompt = f"{base_prompt}, {variation_prompt}"
            
            # Generate new variation
            variation = model_handler.generate_image(
                prompt=full_prompt,
                reference_image=base_image,
                strength=0.75
            )
            
            # Save new variation
            variation_path = Path(metadata["training_images"][image_index])
            variation.save(variation_path)
            
            metadata["last_modified"] = str(datetime.datetime.now())
            self._save_metadata(character_id, metadata)
            return metadata
            
        except IndexError:
            raise ValueError(f"Image index {image_index} not found") 