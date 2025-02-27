from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from model_handler import StableDiffusionHandler
from character_manager import CharacterManager, CharacterState
from config import ModelConfig, GenerationConfig, APIConfig
import tempfile
import os
import logging
from typing import Optional, Dict, List, Union, Annotated
from pathlib import Path
import uuid
from pydantic import BaseModel
from moviepy.editor import ImageSequenceClip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    logger.info("Initializing services...")
    model_config = ModelConfig()
    logger.info(f"Model config: {model_config}")
    model = StableDiffusionHandler(model_config)
    character_manager = CharacterManager(model_config)
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    model = None
    character_manager = None

class CreateCharacterRequest(BaseModel):
    prompt: str
    regenerate: bool = False
    character_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    negative_prompt: Optional[str] = None
    use_default_prompt_enhancements: Optional[bool] = None
    custom_prompt_modifiers: Optional[List[str]] = None
    use_default_negative_enhancements: Optional[bool] = None
    custom_negative_modifiers: Optional[List[str]] = None

class GenerateTrainingRequest(BaseModel):
    num_variations: int = 10
    custom_prompts: Optional[List[str]] = None

class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    guidance_scale: Optional[float] = None
    character_consistency: Optional[float] = None

async def get_form_data(
    prompt: Annotated[str, Form()],
    negative_prompt: Annotated[Optional[str], Form()] = None,
    regenerate: Annotated[bool, Form()] = False,
    character_id: Annotated[Optional[str], Form()] = None,
    existing_image: Annotated[Optional[UploadFile], File()] = None,
    width: Annotated[Optional[int], Form()] = None,
    height: Annotated[Optional[int], Form()] = None,
    use_default_prompt_enhancements: Annotated[Optional[bool], Form()] = None,
    custom_prompt_modifiers: Annotated[Optional[str], Form()] = None,
    use_default_negative_enhancements: Annotated[Optional[bool], Form()] = None,
    custom_negative_modifiers: Annotated[Optional[str], Form()] = None
) -> dict:
    # Parse comma-separated strings into lists if provided
    custom_prompt_list = None
    if custom_prompt_modifiers:
        custom_prompt_list = [item.strip() for item in custom_prompt_modifiers.split(',')]
        
    custom_negative_list = None
    if custom_negative_modifiers:
        custom_negative_list = [item.strip() for item in custom_negative_modifiers.split(',')]
    
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "regenerate": regenerate,
        "character_id": character_id,
        "existing_image": existing_image,
        "width": width,
        "height": height,
        "use_default_prompt_enhancements": use_default_prompt_enhancements,
        "custom_prompt_modifiers": custom_prompt_list,
        "use_default_negative_enhancements": use_default_negative_enhancements,
        "custom_negative_modifiers": custom_negative_list
    }

@app.get("/")
async def root():
    model_status = "initialized" if model is not None else "failed to initialize"
    return {
        "status": "ok" if model is not None else "error",
        "message": f"Character Generator API is running. Model status: {model_status}"
    }

@app.get("/health")
async def health_check():
    if model is None or character_manager is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Services failed to initialize"}
        )
    return {"status": "healthy"}

@app.post("/characters/create_initial")
async def create_initial_character(
    form_data: Annotated[dict, Depends(get_form_data)]
) -> Dict:
    """Create a new character from either a description or existing image"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")

        # Validate inputs
        if not form_data["prompt"]:
            raise HTTPException(
                status_code=422,
                detail="A valid prompt string is required"
            )
            
        if form_data["regenerate"] and not form_data["character_id"]:
            raise HTTPException(
                status_code=422,
                detail="character_id is required when regenerate=True"
            )

        # Handle existing image if provided
        image = None
        if form_data["existing_image"]:
            try:
                contents = await form_data["existing_image"].read()
                image = Image.open(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to process uploaded image: {str(e)}"
                )
            
        if form_data["regenerate"]:
            # Generate new version for existing character
            try:
                # Get existing metadata to preserve dimensions if not specified
                existing_metadata = character_manager.get_character_info(form_data["character_id"])
                
                # Create generation config with custom dimensions if provided
                # or use existing dimensions if available
                gen_config = GenerationConfig(
                    width=form_data.get("width", existing_metadata.get("width", 1024)),
                    height=form_data.get("height", existing_metadata.get("height", 1024))
                )
                
                # Add prompt enhancement options if provided
                if form_data.get("use_default_prompt_enhancements") is not None:
                    gen_config.use_default_prompt_enhancements = form_data["use_default_prompt_enhancements"]
                
                if form_data.get("custom_prompt_modifiers") is not None:
                    gen_config.custom_prompt_modifiers = form_data["custom_prompt_modifiers"]
                    
                if form_data.get("use_default_negative_enhancements") is not None:
                    gen_config.use_default_negative_enhancements = form_data["use_default_negative_enhancements"]
                    
                if form_data.get("custom_negative_modifiers") is not None:
                    gen_config.custom_negative_modifiers = form_data["custom_negative_modifiers"]
                
                generated = model.generate_image(
                    prompt=form_data["prompt"],
                    negative_prompt=form_data.get("negative_prompt"),
                    gen_config=gen_config
                )
                metadata = character_manager.save_generated_base(form_data["character_id"], generated)
                
                # Store the dimensions and prompt enhancement settings in metadata
                metadata["width"] = gen_config.width
                metadata["height"] = gen_config.height
                
                # Store prompt enhancement settings if they were customized
                if form_data.get("use_default_prompt_enhancements") is not None:
                    metadata["use_default_prompt_enhancements"] = gen_config.use_default_prompt_enhancements
                    
                if form_data.get("custom_prompt_modifiers") is not None:
                    metadata["custom_prompt_modifiers"] = gen_config.custom_prompt_modifiers
                    
                if form_data.get("use_default_negative_enhancements") is not None:
                    metadata["use_default_negative_enhancements"] = gen_config.use_default_negative_enhancements
                    
                if form_data.get("custom_negative_modifiers") is not None:
                    metadata["custom_negative_modifiers"] = gen_config.custom_negative_modifiers
                    
                character_manager._save_metadata(metadata["id"], metadata)
                
                return {
                    "status": "regenerated",
                    "character": metadata,
                    "message": "New base image generated. Use /approve endpoint when satisfied."
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        else:
            # Create new character
            try:
                if image:
                    # Use uploaded image
                    metadata = character_manager.create_initial_character(form_data["prompt"], image)
                    return {
                        "status": "created",
                        "character": metadata,
                        "message": "Character created with uploaded image"
                    }
                else:
                    # Generate initial image
                    metadata = character_manager.create_initial_character(form_data["prompt"])
                    
                    # Create generation config with custom dimensions if provided
                    gen_config = GenerationConfig(
                        width=form_data.get("width", 1024),
                        height=form_data.get("height", 1024)
                    )
                    
                    # Add prompt enhancement options if provided
                    if form_data.get("use_default_prompt_enhancements") is not None:
                        gen_config.use_default_prompt_enhancements = form_data["use_default_prompt_enhancements"]
                    
                    if form_data.get("custom_prompt_modifiers") is not None:
                        gen_config.custom_prompt_modifiers = form_data["custom_prompt_modifiers"]
                        
                    if form_data.get("use_default_negative_enhancements") is not None:
                        gen_config.use_default_negative_enhancements = form_data["use_default_negative_enhancements"]
                        
                    if form_data.get("custom_negative_modifiers") is not None:
                        gen_config.custom_negative_modifiers = form_data["custom_negative_modifiers"]
                    
                    generated = model.generate_image(
                        prompt=form_data["prompt"],
                        negative_prompt=form_data.get("negative_prompt"),
                        gen_config=gen_config
                    )
                    metadata = character_manager.save_generated_base(metadata["id"], generated)
                    
                    # Store the dimensions and prompt enhancement settings in metadata
                    metadata["width"] = gen_config.width
                    metadata["height"] = gen_config.height
                    
                    # Store prompt enhancement settings if they were customized
                    if form_data.get("use_default_prompt_enhancements") is not None:
                        metadata["use_default_prompt_enhancements"] = gen_config.use_default_prompt_enhancements
                        
                    if form_data.get("custom_prompt_modifiers") is not None:
                        metadata["custom_prompt_modifiers"] = gen_config.custom_prompt_modifiers
                        
                    if form_data.get("use_default_negative_enhancements") is not None:
                        metadata["use_default_negative_enhancements"] = gen_config.use_default_negative_enhancements
                        
                    if form_data.get("custom_negative_modifiers") is not None:
                        metadata["custom_negative_modifiers"] = gen_config.custom_negative_modifiers
                        
                    character_manager._save_metadata(metadata["id"], metadata)
                    
                    return {
                        "status": "created",
                        "character": metadata,
                        "message": "Character created with generated image. Use /approve endpoint when satisfied."
                    }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create character: {str(e)}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_initial_character: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/characters/{character_id}/approve")
async def approve_character_base(
    character_id: str,
    use_latest: bool = True
) -> Dict:
    """Approve the current base image for a character"""
    try:
        if character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        metadata = character_manager.approve_base_image(character_id, use_latest)
        return {
            "status": "approved",
            "character": metadata,
            "message": "Base image approved. Ready for training data generation."
        }
        
    except Exception as e:
        logger.error(f"Error in approve_character_base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/characters/{character_id}")
async def get_character_info(character_id: str) -> Dict:
    """Get current character information and state"""
    try:
        if character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        return character_manager.get_character_info(character_id)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_character_info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/characters/{character_id}/base_image")
async def get_base_image(character_id: str):
    """Get the character's base image"""
    try:
        if character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        metadata = character_manager.get_character_info(character_id)
        
        # Get the appropriate image path
        if "latest_generated_base" in metadata:
            image_path = metadata["latest_generated_base"]
        elif "base_image_path" in metadata:
            image_path = metadata["base_image_path"]
        else:
            raise HTTPException(status_code=404, detail="No base image found for this character")
            
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
            
        return FileResponse(
            image_path,
            media_type="image/png",
            filename=f"character_{character_id}_base.png"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_base_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    character_id: Optional[str] = None,
    reference_image: UploadFile = File(None),
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.8,
    is_base_character: bool = False
):
    """
    @deprecated Use /characters/create_initial for new characters
    and /characters/{id}/generate for existing ones
    """
    try:
        if model is None or character_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Services not initialized. Please check server logs."
            )
            
        # Handle reference image if provided
        reference_latents = None
        if reference_image:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await reference_image.read())
                reference_latents = model.encode_reference_image(temp_file.name)
            os.unlink(temp_file.name)
            
        # Generate image
        gen_config = GenerationConfig(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            character_id=character_id
        )
        
        # If using a character LoRA, load it
        if character_id:
            lora_path = character_manager.get_character_lora_path(character_id)
            model.load_character_lora(lora_path)
        
        generated_image = model.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_latents=reference_latents,
            gen_config=gen_config
        )
        
        # If this is a base character, save it and train LoRA
        if is_base_character:
            character_id = character_manager.save_base_character(generated_image, prompt)
            return {
                "message": "Base character created successfully",
                "character_id": character_id
            }
        
        # For regular generations, just return the image
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return FileResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters/{character_id}/generate_training")
async def generate_training_data(
    character_id: str,
    request: GenerateTrainingRequest
) -> Dict:
    """Generate training data variations for a character"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        metadata = character_manager.generate_training_data(
            character_id=character_id,
            num_variations=request.num_variations,
            custom_prompts=request.custom_prompts,
            model_handler=model
        )
        
        return {
            "status": "completed",
            "character": metadata,
            "message": f"Generated {len(metadata['training_images'])} training images"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in generate_training_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/characters/{character_id}/training/{image_index}")
async def get_training_image(character_id: str, image_index: int):
    """Get a specific training image"""
    try:
        if character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        image_path = character_manager.get_training_image(character_id, image_index)
        if not image_path:
            raise HTTPException(status_code=404, detail="Training image not found")
            
        return FileResponse(
            image_path,
            media_type="image/png",
            filename=f"training_{image_index}.png"
        )
        
    except Exception as e:
        logger.error(f"Error in get_training_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/characters/{character_id}/training/{image_index}")
async def remove_training_image(character_id: str, image_index: int) -> Dict:
    """Remove a training image"""
    try:
        if character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        metadata = character_manager.remove_training_image(character_id, image_index)
        return {
            "status": "removed",
            "character": metadata,
            "message": f"Removed training image {image_index}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in remove_training_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters/{character_id}/training/{image_index}/regenerate")
async def regenerate_training_image(
    character_id: str,
    image_index: int,
    custom_prompt: Optional[str] = None
) -> Dict:
    """Regenerate a specific training image"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        metadata = character_manager.regenerate_training_image(
            character_id=character_id,
            image_index=image_index,
            model_handler=model,
            custom_prompt=custom_prompt
        )
        
        return {
            "status": "regenerated",
            "character": metadata,
            "message": f"Regenerated training image {image_index}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in regenerate_training_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters/{character_id}/train")
async def start_training(
    character_id: str,
    background_tasks: BackgroundTasks
) -> Dict:
    """Start LoRA training for a character"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        # Start training in background
        metadata = character_manager.start_training(character_id, model)
        
        return {
            "status": "started",
            "character": metadata,
            "message": "Training started. Check status with /training_status endpoint."
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in start_training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/characters/{character_id}/training_status")
async def get_training_status(character_id: str) -> Dict:
    """Get current training status"""
    try:
        if character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        status = character_manager.get_training_status(character_id)
        return {
            "status": "success",
            "training_status": status
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_training_status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters/{character_id}/generate_scene")
async def generate_scene(
    character_id: str,
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5
) -> FileResponse:
    """Generate a new scene with the character using their LoRA model"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        # Check character status
        status = character_manager.get_training_status(character_id)
        if status["state"] != CharacterState.READY:
            raise HTTPException(
                status_code=400,
                detail="Character LoRA model not ready. Complete training first."
            )
            
        # Load character's LoRA
        lora_path = character_manager.get_character_lora_path(character_id)
        model.load_character_lora(lora_path)
        
        # Generate scene
        gen_config = GenerationConfig(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        generated_image = model.generate_image(
            prompt=prompt,
            gen_config=gen_config
        )
        
        # Save and return image
        output_path = Path(character_manager.storage_config.output_path)
        os.makedirs(output_path, exist_ok=True)
        image_path = output_path / f"{character_id}_{uuid.uuid4()}.png"
        generated_image.save(image_path)
        
        return FileResponse(
            image_path,
            media_type="image/png",
            filename=f"scene.png"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in generate_scene: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters/{character_id}/train/resume")
async def resume_training(
    character_id: str,
    background_tasks: BackgroundTasks,
    checkpoint_path: Optional[str] = None
) -> Dict:
    """Resume LoRA training from the latest checkpoint"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        metadata = character_manager.resume_training(
            character_id=character_id,
            model_handler=model,
            checkpoint_path=checkpoint_path
        )
        
        return {
            "status": "resumed",
            "character": metadata,
            "message": "Training resumed. Check status with /training_status endpoint."
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in resume_training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters/{character_id}/generate_video")
async def generate_character_video(
    character_id: str,
    request: VideoGenerationRequest
) -> FileResponse:
    """Generate a video featuring the character in a scene"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        # Load character info and validate state
        metadata = character_manager.get_character_info(character_id)
        if "base_image_path" not in metadata or not metadata.get("base_image_approved"):
            raise HTTPException(
                status_code=400,
                detail="Character must have an approved base image"
            )
            
        # Get video generation config with overrides
        video_config = model.config.video
        if request.fps:
            video_config.fps = request.fps
        if request.num_frames:
            video_config.num_frames = request.num_frames
        if request.width:
            video_config.width = request.width
        if request.height:
            video_config.height = request.height
        if request.guidance_scale:
            video_config.guidance_scale = request.guidance_scale
        if request.character_consistency is not None:
            video_config.character_consistency = request.character_consistency
            
        # Load and encode reference image
        reference_latents = model.encode_reference_image(metadata["base_image_path"])
        
        # Generate video frames
        logger.info("Generating video frames...")
        frames = model.generate_video(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            reference_latents=reference_latents,
            num_frames=video_config.num_frames,
            width=video_config.width,
            height=video_config.height,
            guidance_scale=video_config.guidance_scale,
            character_consistency=video_config.character_consistency
        )
        
        # Create video file
        logger.info("Creating video file...")
        clip = ImageSequenceClip(frames, fps=video_config.fps)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            clip.write_videofile(
                temp_file.name,
                codec='libx264',
                audio=False
            )
            
            # Move to output directory
            output_dir = Path(model.config.storage.output_path)
            video_path = output_dir / f"{character_id}_{uuid.uuid4()}.mp4"
            os.rename(temp_file.name, video_path)
            
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"character_scene.mp4"
        )
        
    except Exception as e:
        logger.error(f"Failed to generate video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/characters/create")
async def create_character_json(request: CreateCharacterRequest) -> Dict:
    """Create a new character from a JSON request"""
    try:
        if model is None or character_manager is None:
            raise HTTPException(status_code=503, detail="Services not initialized")

        # Convert request to form_data format for reuse
        form_data = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "regenerate": request.regenerate,
            "character_id": request.character_id,
            "existing_image": None,  # JSON endpoint doesn't support file uploads
            "width": request.width,
            "height": request.height
        }
        
        if request.regenerate:
            # Generate new version for existing character
            try:
                # Get existing metadata to preserve dimensions if not specified
                existing_metadata = character_manager.get_character_info(request.character_id)
                
                # Create generation config with custom dimensions if provided
                # or use existing dimensions if available
                gen_config = GenerationConfig(
                    width=request.width or existing_metadata.get("width", 1024),
                    height=request.height or existing_metadata.get("height", 1024)
                )
                
                # Add prompt enhancement options if provided
                if request.use_default_prompt_enhancements is not None:
                    gen_config.use_default_prompt_enhancements = request.use_default_prompt_enhancements
                
                if request.custom_prompt_modifiers is not None:
                    gen_config.custom_prompt_modifiers = request.custom_prompt_modifiers
                    
                if request.use_default_negative_enhancements is not None:
                    gen_config.use_default_negative_enhancements = request.use_default_negative_enhancements
                    
                if request.custom_negative_modifiers is not None:
                    gen_config.custom_negative_modifiers = request.custom_negative_modifiers
                
                generated = model.generate_image(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    gen_config=gen_config
                )
                metadata = character_manager.save_generated_base(request.character_id, generated)
                
                # Store the dimensions and prompt enhancement settings in metadata
                metadata["width"] = gen_config.width
                metadata["height"] = gen_config.height
                
                # Store prompt enhancement settings if they were customized
                if request.use_default_prompt_enhancements is not None:
                    metadata["use_default_prompt_enhancements"] = gen_config.use_default_prompt_enhancements
                    
                if request.custom_prompt_modifiers is not None:
                    metadata["custom_prompt_modifiers"] = gen_config.custom_prompt_modifiers
                    
                if request.use_default_negative_enhancements is not None:
                    metadata["use_default_negative_enhancements"] = gen_config.use_default_negative_enhancements
                    
                if request.custom_negative_modifiers is not None:
                    metadata["custom_negative_modifiers"] = gen_config.custom_negative_modifiers
                    
                character_manager._save_metadata(metadata["id"], metadata)
                
                return {
                    "status": "regenerated",
                    "character": metadata,
                    "message": "New base image generated. Use /approve endpoint when satisfied."
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        else:
            # Create new character
            try:
                # Generate initial image
                metadata = character_manager.create_initial_character(request.prompt)
                
                # Create generation config with custom dimensions if provided
                gen_config = GenerationConfig(
                    width=request.width or 1024,
                    height=request.height or 1024
                )
                
                # Add prompt enhancement options if provided
                if request.use_default_prompt_enhancements is not None:
                    gen_config.use_default_prompt_enhancements = request.use_default_prompt_enhancements
                
                if request.custom_prompt_modifiers is not None:
                    gen_config.custom_prompt_modifiers = request.custom_prompt_modifiers
                    
                if request.use_default_negative_enhancements is not None:
                    gen_config.use_default_negative_enhancements = request.use_default_negative_enhancements
                    
                if request.custom_negative_modifiers is not None:
                    gen_config.custom_negative_modifiers = request.custom_negative_modifiers
                    
                generated = model.generate_image(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    gen_config=gen_config
                )
                metadata = character_manager.save_generated_base(metadata["id"], generated)
                
                # Store the dimensions and prompt enhancement settings in metadata
                metadata["width"] = gen_config.width
                metadata["height"] = gen_config.height
                
                # Store prompt enhancement settings if they were customized
                if request.use_default_prompt_enhancements is not None:
                    metadata["use_default_prompt_enhancements"] = gen_config.use_default_prompt_enhancements
                    
                if request.custom_prompt_modifiers is not None:
                    metadata["custom_prompt_modifiers"] = gen_config.custom_prompt_modifiers
                    
                if request.use_default_negative_enhancements is not None:
                    metadata["use_default_negative_enhancements"] = gen_config.use_default_negative_enhancements
                    
                if request.custom_negative_modifiers is not None:
                    metadata["custom_negative_modifiers"] = gen_config.custom_negative_modifiers
                    
                character_manager._save_metadata(metadata["id"], metadata)
                
                return {
                    "status": "created",
                    "character": metadata,
                    "message": "Character created with generated image. Use /approve endpoint when satisfied."
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create character: {str(e)}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_character_json: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    ) 