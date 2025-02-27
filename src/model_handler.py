import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from typing import Optional, List
from config import ModelConfig, GenerationConfig
import logging
import os
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import OpenSora, but don't fail if not available
try:
    from opensora.opensora import OpenSoraPipeline, WFVAEModel
    OPENSORA_AVAILABLE = True
except ImportError:
    logger.warning("OpenSora not available. Video generation features will be disabled.")
    OPENSORA_AVAILABLE = False

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

        # Initialize video components as None
        self.video_pipeline = None
        self.video_vae = None
        
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
            
            # Initialize video generation pipeline only if OpenSora is available
            if OPENSORA_AVAILABLE and hasattr(model_config, 'opensora_path'):
                self._init_video_pipeline()
                
        except Exception as e:
            logger.error(f"Failed to initialize StableDiffusionHandler: {str(e)}")
            raise

    def _init_video_pipeline(self):
        """Initialize the video generation pipeline if OpenSora is available"""
        try:
            logger.info("Initializing video generation pipeline...")
            self.video_vae = WFVAEModel.from_pretrained(
                f"{self.config.opensora_path}/vae",
                torch_dtype=self.dtype
            ).to(self.device)
            
            self.video_pipeline = OpenSoraPipeline.from_pretrained(
                self.config.opensora_path,
                vae=self.video_vae,
                torch_dtype=self.dtype
            ).to(self.device)
            logger.info("Video pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Open-Sora pipeline: {str(e)}")
            self.video_pipeline = None
            self.video_vae = None
        
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
        
    def _enhance_prompt(self, prompt: str, gen_config: Optional[GenerationConfig] = None) -> str:
        """Add modifiers to the user prompt based on configuration."""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
            
        # If no config provided or default enhancements are disabled, return the prompt as is
        if gen_config is None:
            gen_config = GenerationConfig()
            
        # Extract the core character description (use the first sentence)
        main_desc = prompt.split(".")[0].strip()
        if not main_desc.endswith("."):
            main_desc += "."
            
        # If default enhancements are disabled and no custom modifiers provided, return just the main description
        if not gen_config.use_default_prompt_enhancements and not gen_config.custom_prompt_modifiers:
            logger.info(f"Using prompt without enhancements: {prompt}")
            return prompt
            
        # Determine which modifiers to use
        if not gen_config.use_default_prompt_enhancements and gen_config.custom_prompt_modifiers:
            # Use custom modifiers
            modifiers = gen_config.custom_prompt_modifiers
            logger.info(f"Using custom prompt modifiers: {modifiers}")
        else:
            # Use default modifiers
            modifiers = [
                "single character only",
                "full body from head to toe",
                "full body, standing straight ahead",
                "arms extended out to the sides in a Vitruvian pose",
                "symmetrical, one set of arms and legs",
                "pure white background",
                "professional photography",
                "photorealistic"
            ]
            
        # Combine the main description with the modifiers
        enhanced_prompt = f"{main_desc}, {', '.join(modifiers)}"
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt


    def _enhance_negative_prompt(self, negative_prompt: Optional[str] = None, gen_config: Optional[GenerationConfig] = None) -> str:
        """Combine user negative prompt with modifiers based on configuration."""
        if gen_config is None:
            gen_config = GenerationConfig()
            
        # If default enhancements are disabled and no custom modifiers provided, return the negative prompt as is
        if not gen_config.use_default_negative_enhancements and not gen_config.custom_negative_modifiers:
            return negative_prompt or ""
            
        # Determine which negative modifiers to use
        if not gen_config.use_default_negative_enhancements and gen_config.custom_negative_modifiers:
            # Use custom negative modifiers
            base_negative = gen_config.custom_negative_modifiers
            logger.info(f"Using custom negative modifiers: {base_negative}")
        else:
            # Use default negative modifiers
            base_negative = [
                "multiple characters",
                "multiple views",
                "split image",
                "side by side",
                "duplicate",
                "multiple versions",
                "abstract",
                "blurry",
                "bad quality",
                "worst quality",
                "low quality",
                "normal quality",
                "lowres",
                "distorted",
                "deformed",
                "mutation",
                "extra limbs",
                "missing limbs",
                "disconnected limbs",
                "malformed",
                "out of frame",
                "cropped",
                "watermark",
                "signature",
                "text",
                "error",
                "jpeg artifacts",
                "duplicate characters"
            ]
        
        if negative_prompt:
            return f"{negative_prompt}, {', '.join(base_negative)}"
        return ', '.join(base_negative)


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
            enhanced_prompt = self._enhance_prompt(prompt, gen_config)
            enhanced_negative_prompt = self._enhance_negative_prompt(negative_prompt, gen_config)
            
            # Get width and height from config, ensuring they're multiples of 8
            width = getattr(gen_config, 'width', 1024)
            height = getattr(gen_config, 'height', 1024)
            
            # Ensure dimensions are multiples of 8 as required by Stable Diffusion
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            # Updated generation parameters for better quality
            pipeline_kwargs = {
                "prompt": enhanced_prompt,
                "negative_prompt": enhanced_negative_prompt,
                "num_inference_steps": getattr(gen_config, 'num_inference_steps', 50),  # Increased steps
                "guidance_scale": getattr(gen_config, 'guidance_scale', 8.5),  # Adjusted guidance
                "width": width,
                "height": height,
                "latents": reference_latents if reference_latents is not None else None,
            }
            
            # Generate the image
            with torch.no_grad():
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

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        reference_latents: Optional[torch.Tensor] = None,
        num_frames: int = 93,
        width: int = 640,
        height: int = 640,
        guidance_scale: float = 8.5,
        character_consistency: float = 0.8
    ) -> List[Image.Image]:
        """Generate video using Open-Sora"""
        try:
            if not OPENSORA_AVAILABLE:
                raise ValueError("OpenSora is not available. Video generation is disabled.")
                
            if self.video_pipeline is None:
                raise ValueError("Video pipeline is not initialized. Please check OpenSora installation.")

            # Enhance prompt for video context
            enhanced_prompt = self._enhance_prompt(prompt)
            enhanced_negative = self._enhance_negative_prompt(negative_prompt)
            
            logger.info(f"Generating video with prompt: {enhanced_prompt}")
            
            # Generate video frames
            frames = self.video_pipeline(
                enhanced_prompt,
                negative_prompt=enhanced_negative,
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=50
            ).frames
            
            # If we have reference latents, apply character consistency
            if reference_latents is not None and character_consistency > 0:
                frames = self._apply_character_consistency(
                    frames,
                    reference_latents,
                    character_consistency
                )
                
            return frames
            
        except Exception as e:
            logger.error(f"Failed to generate video: {str(e)}")
            raise

    def _apply_character_consistency(
        self,
        frames: List[Image.Image],
        reference_latents: torch.Tensor,
        consistency_weight: float
    ) -> List[Image.Image]:
        """Apply character consistency to generated frames"""
        try:
            # Encode frames to latent space
            frame_latents = torch.stack([
                self.vae.encode(self._preprocess_image(frame)).latent_dist.sample()
                for frame in frames
            ]).to(self.device)
            
            # Interpolate reference latents to match frame dimensions
            ref_latents = F.interpolate(
                reference_latents.unsqueeze(0).repeat(len(frames), 1, 1, 1),
                size=frame_latents.shape[2:],
                mode='bilinear'
            )
            
            # Blend latents
            blended_latents = (
                frame_latents * (1 - consistency_weight) +
                ref_latents * consistency_weight
            )
            
            # Decode back to images
            return [
                Image.fromarray(
                    self.vae.decode(latent).sample.cpu().numpy().astype(np.uint8)
                )
                for latent in blended_latents
            ]
            
        except Exception as e:
            logger.error(f"Failed to apply character consistency: {str(e)}")
            raise

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        return torch.from_numpy(
            np.array(image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device) 