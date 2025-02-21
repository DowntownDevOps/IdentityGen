import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from typing import Optional
from config import ModelConfig, GenerationConfig
import logging
import os

logger = logging.getLogger(__name__)

class StableDiffusionHandler:
    def __init__(self, model_config: ModelConfig):
        logger.info("Initializing StableDiffusionHandler...")
        self.config = model_config
        self.device = torch.device(model_config.device)
        self.dtype = getattr(torch, model_config.dtype)
        
        # Default prompt modifiers for consistent character generation
        self._default_prompt_modifiers = [
            "full body from head to feet",
            "full length character turnaround reference",
            "standing perfectly straight",
            "feet firmly planted shoulder width apart",
            "arms relaxed at sides",
            "palms facing thighs",
            "facing directly at viewer",
            "perfectly centered in frame",
            "neutral stance",
            "neutral facial expression",
            "pure white void background",
            "no background elements",
            "no decorative elements",
            "no frame",
            "no borders",
            "studio reference lighting",
            "clear view of entire body",
            "shows complete feet",
            "shows complete hands",
            "professional 3D model reference",
            "high detail",
            "masterpiece",
            "sharp focus",
            "8k uhd",
            "photorealistic",
            "RAW candid cinema",
            "16mm, color graded Portra 400 film",
            "remarkable color",
            "ultra-realistic",
            "textured skin",
            "remarkably detailed pupils",
            "realistic dull skin noise",
            "visible skin detail",
            "skin fuzz",
            "dry skin",
            "shot with cinematic camera",
            "50mm lens perspective",
            "natural light from the right creating a soft glow",
            "subject's eyes aligned with the rule of thirds",
            "slight exposure adjustment for brightness",
            "warm, cheerful atmosphere"
        ]

        self._default_negative_modifiers = [
            "cropped",
            "close up",
            "portrait",
            "zoomed in",
            "partial body",
            "cut off feet",
            "cut off legs",
            "missing feet",
            "missing hands",
            "missing limbs",
            "bad anatomy",
            "extra limbs",
            "floating limbs",
            "disconnected limbs",
            "malformed hands",
            "malformed feet",
            "mutated",
            "deformed",
            "blurry",
            "duplicate",
            "watermark",
            "signature",
            "text",
            "frame",
            "border",
            "background pattern",
            "background texture",
            "background design",
            "decorative elements",
            "props",
            "weapons",
            "accessories",
            "pets",
            "animals",
            "scenery",
            "environment",
            "dynamic pose",
            "action pose",
            "tilted head",
            "twisted body",
            "artistic background",
            "gradient background",
            "textured background",
            "doll",
            "anime",
            "animation",
            "cartoon",
            "render",
            "artwork",
            "semi-realistic",
            "CGI",
            "3d",
            "sketch",
            "drawing"
        ]


        try:
            logger.info(f"Loading pipeline from {model_config.base_model_path}")
            # Initialize base pipeline with safety checker disabled for training
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_config.base_model_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checking=False
            ).to(self.device)
            
            # Enable memory efficient attention
            self.pipe.enable_attention_slicing()
            
            # Load LoRA if specified
            if model_config.lora_path:
                self.load_character_lora(model_config.lora_path)
                
            logger.info("Initializing VAE and CLIP components")
            # Initialize VAE and CLIP components from SDXL
            self.vae = self.pipe.vae
            self.vae.to(device=self.device, dtype=self.dtype)
            
            # Use CLIP components from SDXL pipeline
            self.tokenizer = self.pipe.tokenizer
            self.text_encoder = self.pipe.text_encoder
            self.text_encoder.to(device=self.device, dtype=self.dtype)
            
            logger.info("StableDiffusionHandler initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize StableDiffusionHandler: {str(e)}")
            raise
        
    def load_character_lora(self, lora_path: str):
        """Load a character-specific LoRA model"""
        logger.info(f"Loading character LoRA from {lora_path}")
        try:
            self.pipe.unet.load_attn_procs(lora_path)
            logger.info("Character LoRA loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load character LoRA: {str(e)}")
            raise
        
    def encode_reference_image(self, image_path: str) -> torch.Tensor:
        """Convert reference image to latent representation"""
        try:
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("Invalid image path provided")
                
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # SDXL expects 1024x1024 images
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Convert to tensor with correct shape and type
            image_tensor = torch.from_numpy(image_array)
            image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
            image_tensor = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
            
            # Encode to latents
            with torch.no_grad():
                # Scale down latents to match SDXL's expected size
                latents = self.vae.encode(image_tensor).latent_dist.sample()
                latents = torch.nn.functional.interpolate(
                    latents,
                    size=(64, 64),  # SDXL expects 64x64 latents
                    mode='bilinear',
                    align_corners=False
                )
                latents = latents * 0.18215
                
            if latents.dtype != self.dtype:
                latents = latents.to(dtype=self.dtype)
                
            return latents
            
        except Exception as e:
            logger.error(f"Failed to encode reference image: {str(e)}")
            raise
        
    def _enhance_prompt(self, prompt: str) -> str:
        """Add default modifiers to user prompt for consistent character generation"""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
            
        # For training prompts that already include view/pose instructions, don't add modifiers
        if any(keyword in prompt.lower() for keyword in ["front view", "side view", "back view", "3/4 view"]):
            enhanced_prompt = f"{prompt}, RAW photo, photorealistic, sharp focus"
        else:
            # Extract the first sentence of the prompt (main character description)
            main_desc = prompt.split(".")[0] + "."
            
            # Focused set of modifiers for photorealistic full-body shots
            essential_modifiers = [
                "full body centered",
                "standing pose",
                "white background",
                "studio lighting",
                "RAW photo",
                "photorealistic",
                "sharp focus",
                "professional photography",
                "4k"
            ]
            
            # Combine main description with essential modifiers
            enhanced_prompt = f"{main_desc}, {', '.join(essential_modifiers)}"
            
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt

    def _enhance_negative_prompt(self, negative_prompt: Optional[str] = None) -> str:
        """Combine user negative prompt with default negative modifiers"""
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError("Negative prompt must be a string if provided")
            
        # Select essential negative modifiers
        essential_negative = [
            "cropped",
            "close up",
            "portrait",
            "zoomed in",
            "partial body",
            "cut off feet",
            "cut off legs",
            "missing feet",
            "missing hands",
            "missing limbs",
            "bad anatomy",
            "extra limbs",
            "floating limbs",
            "disconnected limbs",
            "malformed hands",
            "malformed feet",
            "mutated",
            "deformed",
            "blurry",
            "duplicate",
            "watermark",
            "signature",
            "text",
            "frame",
            "border",
            "background pattern",
            "background texture",
            "background design",
            "decorative elements",
            "props",
            "weapons",
            "accessories",
            "pets",
            "animals",
            "scenery",
            "environment",
            "dynamic pose",
            "action pose",
            "tilted head",
            "twisted body",
            "artistic background",
            "gradient background",
            "textured background",
            "doll",
            "anime",
            "animation",
            "cartoon",
            "render",
            "artwork",
            "semi-realistic",
            "CGI",
            "3d",
            "sketch",
            "drawing"
        ]
        
        base_negative = ", ".join(essential_negative)
        if negative_prompt:
            return f"{negative_prompt}, {base_negative}"
        return base_negative

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        reference_latents: Optional[torch.Tensor] = None,
        gen_config: Optional[GenerationConfig] = None
    ) -> Image.Image:
        """Generate new image based on prompt and optional reference"""
        try:
            if gen_config is None:
                gen_config = GenerationConfig()
                
            # Validate inputs
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Invalid prompt provided")
                
            # Enhance prompts with default modifiers
            enhanced_prompt = self._enhance_prompt(prompt)
            enhanced_negative_prompt = self._enhance_negative_prompt(negative_prompt)
                
            # Generate base latents if no reference provided
            if reference_latents is None:
                latents = torch.randn(
                    (1, 4, 64, 64),
                    device=self.device,
                    dtype=self.dtype
                )
            else:
                # Validate reference latents
                if not isinstance(reference_latents, torch.Tensor):
                    raise ValueError("Reference latents must be a torch.Tensor")
                    
                if reference_latents.shape != (1, 4, 64, 64):
                    raise ValueError(f"Invalid reference latents shape: {reference_latents.shape}, expected (1, 4, 64, 64)")
                    
                # Ensure reference latents are on correct device and dtype
                reference_latents = reference_latents.to(device=self.device, dtype=self.dtype)
                
                # Get random seed from config or generate one
                seed = getattr(gen_config, 'seed', None)
                if seed is None:
                    seed = torch.randint(0, 2**32 - 1, (1,)).item()
                
                # Create generator with random seed
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                # Get strength from config (default to lower value for better consistency)
                strength = getattr(gen_config, 'strength', 0.6)
                
                # Initialize scheduler timesteps
                num_inference_steps = getattr(gen_config, 'num_inference_steps', 30)
                self.pipe.scheduler.set_timesteps(num_inference_steps)
                
                # Generate random noise
                noise = torch.randn(
                    reference_latents.shape,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype
                )
                
                # Scale noise to match latent scale
                noise = noise * 0.18215
                
                # Interpolate between reference and noise
                latents = reference_latents * (1 - strength) + noise * strength
            
            # Generate image
            with torch.no_grad():
                # Create pipeline kwargs with adjusted parameters
                pipeline_kwargs = {
                    "prompt": enhanced_prompt,
                    "negative_prompt": enhanced_negative_prompt,
                    "num_inference_steps": getattr(gen_config, 'num_inference_steps', 30),
                    "guidance_scale": getattr(gen_config, 'guidance_scale', 7.5),
                    "latents": latents,
                    "generator": generator if reference_latents is not None else None
                }
                
                # Generate the image
                output = self.pipe(
                    **pipeline_kwargs,
                    output_type="pil",
                    return_dict=True
                )
                
                if not output.images:
                    raise ValueError("Failed to generate image - no output received")
                
                return output.images[0]
            
        except Exception as e:
            logger.error(f"Failed to generate image: {str(e)}")
            raise 