# Character Generator API

A FastAPI-based service that generates character images using Stable Diffusion XL. This service provides a simple API endpoint for generating images based on text prompts, with optional reference images for style guidance and LoRA model support for character customization.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- At least 16GB of GPU memory recommended
- At least 20GB of disk space for models

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd character-generator
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install required Python packages
- Download necessary AI models
- Build and start the Docker container

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Model Configuration
MODEL_PATH=stabilityai/stable-diffusion-xl-base-1.0
DEVICE=cuda
DTYPE=float16

# API Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Generation Defaults
NUM_INFERENCE_STEPS=30
GUIDANCE_SCALE=7.5
STRENGTH=0.8

# Storage Paths
BASE_CHARACTERS_PATH=/app/storage/base_characters
LORA_MODELS_PATH=/app/storage/lora_models
OUTPUT_PATH=/app/storage/outputs
```

## 📚 API Usage

### Health Check

The API provides two health check endpoints:

1. Basic health check:
```bash
curl http://localhost:8000/
```

Expected response:
```json
{
    "status": "ok",
    "message": "Character Generator API is running. Model status: initialized"
}
```

2. Detailed health check:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
    "status": "healthy"
}
```

### Character Creation Workflow

The API follows a specific workflow for creating and training characters:

1. **Initial Character Creation**
   - Create from description:
   ```bash
   curl -X POST "http://localhost:8000/characters/create_initial" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "a noble elven warrior with golden armor"}'
   ```
   
   - Create from existing image:
   ```bash
   curl -X POST "http://localhost:8000/characters/create_initial" \
        -F "prompt=a noble elven warrior with golden armor" \
        -F "existing_image=@character.png"
   ```

   - Regenerate if not satisfied:
   ```bash
   curl -X POST "http://localhost:8000/characters/create_initial" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "a noble elven warrior with golden armor",
            "regenerate": true,
            "character_id": "YOUR_CHARACTER_ID"
        }'
   ```

2. **Approve Base Image**
   ```bash
   curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/approve"
   ```

3. **Generate Training Data**
   - Generate default variations:
   ```bash
   curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/generate_training" \
        -H "Content-Type: application/json" \
        -d '{"num_variations": 10}'
   ```

   - Generate with custom prompts:
   ```bash
   curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/generate_training" \
        -H "Content-Type: application/json" \
        -d '{
            "num_variations": 5,
            "custom_prompts": [
                "character in battle pose",
                "character casting a spell",
                "character riding a horse",
                "character in formal attire",
                "character in stealth mode"
            ]
        }'
   ```

4. **Manage Training Images**
   - View a training image:
   ```bash
   curl "http://localhost:8000/characters/YOUR_CHARACTER_ID/training/0" --output training_0.png
   ```

   - Regenerate a specific training image:
   ```bash
   curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/training/0/regenerate" \
        -H "Content-Type: application/json" \
        -d '{"custom_prompt": "character in a different battle pose"}'
   ```

   - Remove unwanted training image:
   ```bash
   curl -X DELETE "http://localhost:8000/characters/YOUR_CHARACTER_ID/training/0"
   ```

5. **Train LoRA Model**
   Once you have a satisfactory set of training images:
   ```bash
   # Start training
   curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/train"
   ```

   Check training status:
   ```bash
   curl "http://localhost:8000/characters/YOUR_CHARACTER_ID/training_status"
   ```

   Example response:
   ```json
   {
       "status": "success",
       "training_status": {
           "state": "training",
           "progress": 45.5,
           "training_start": "2024-03-14T10:30:00",
           "last_update": "2024-03-14T10:35:00"
       }
   }
   ```

6. **Generate New Scenes**
   Once training is complete (state is "ready"), generate new scenes with your character:
   ```bash
   curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/generate_scene" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "the character exploring an ancient temple",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }' \
        --output scene.png
   ```

### Character States

Characters progress through several states during creation and training:
1. `initial` - Just created, no approved base image
2. `base_approved` - Has approved base image, ready for training data generation
3. `generating_training` - Currently generating training data variations
4. `training` - LoRA model training in progress
5. `ready` - LoRA trained and ready for scene generation
6. `error` - Something went wrong (check error message in status)

### Training Configuration

The LoRA training process can be configured through environment variables:

```env
# LoRA Training Configuration
LORA_RANK=16           # Rank of LoRA matrices
LORA_ALPHA=32          # LoRA scaling factor
NUM_TRAIN_EPOCHS=100   # Number of training epochs
```

The training process requires:
- At least 5 training images
- CUDA-capable GPU with sufficient memory
- Training time varies based on number of images and epochs

### Scene Generation Tips

When generating new scenes with a trained character:
1. Always include distinctive features from the original character description
2. Be specific about the scene and character's pose/action
3. Use the same style keywords as in the original description for consistency

Example prompts:
```bash
# Action scene
curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/generate_scene" \
     -d '{"prompt": "the noble elven warrior with golden armor in an epic battle stance, wielding a glowing sword, dramatic lighting"}'

# Portrait scene
curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/generate_scene" \
     -d '{"prompt": "close up portrait of the noble elven warrior with golden armor, serene expression, detailed face features"}'

# Environmental scene
curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/generate_scene" \
     -d '{"prompt": "the noble elven warrior with golden armor standing in a mystical elven forest, ethereal atmosphere"}'
```

### Character Management

The API supports character management using LoRA (Low-Rank Adaptation) models for consistent character generation:

#### Create Character

Create a new character profile with associated LoRA model:

```bash
curl -X POST "http://localhost:8000/characters/create" \
     -H "Content-Type: application/json" \
     -d '{
         "name": "elf_warrior",
         "description": "A noble elven warrior with golden armor",
         "training_images": ["base_image1.png", "base_image2.png"],
         "lora_config": {
             "r": 16,
             "alpha": 32,
             "target_modules": ["q_proj", "v_proj"]
         }
     }'
```

#### Generate with Character

Generate an image using a specific character's LoRA model:

```bash
curl -X POST "http://localhost:8000/characters/generate" \
     -H "Content-Type: application/json" \
     -d '{
         "character_name": "elf_warrior",
         "prompt": "the character in a battle pose",
         "num_inference_steps": 30,
         "guidance_scale": 7.5
     }' \
     --output character.png
```

### Basic Image Generation

#### Basic Generation

Generate an image using just a prompt:

```bash
curl -X POST "http://localhost:8000/generate?prompt=a%20beautiful%20fantasy%20character" \
     --output character.png
```

#### Advanced Generation

Generate with all parameters and a reference image:

```bash
curl -X POST "http://localhost:8000/generate" \
     -F "prompt=a beautiful fantasy character with long flowing hair" \
     -F "reference_image=@reference.png" \
     -F "num_inference_steps=30" \
     -F "guidance_scale=7.5" \
     -F "strength=0.8" \
     --output character.png
```

#### Using Generated Images as References

The API doesn't store generated images - they are saved locally where you make the API call. To use a generated image as a reference:

1. First generate and save an image:
```bash
# Generate first image
curl -X POST "http://localhost:8000/generate?prompt=elf warrior" \
     --output first_character.png
```

2. Then use that saved image as a reference for a new generation:
```bash
# Use first_character.png as reference for new generation
curl -X POST "http://localhost:8000/generate" \
     -F "prompt=elf warrior with different pose" \
     -F "reference_image=@first_character.png" \
     -F "strength=0.8" \
     --output second_character.png
```

The `strength` parameter controls how much influence the reference image has on the final result:
- Higher values (closer to 1.0) preserve more of the reference image's style and composition
- Lower values (closer to 0.0) allow more deviation from the reference

### Python Example

```python
import requests
from PIL import Image
import io

def generate_character(
    prompt: str,
    reference_image_path: str = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.8
):
    url = "http://localhost:8000/generate"
    
    # Prepare parameters
    params = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "strength": strength
    }
    
    # Add reference image if provided
    files = {}
    if reference_image_path:
        files = {"reference_image": open(reference_image_path, "rb")}
    
    # Make request
    response = requests.post(url, params=params, files=files)
    
    if response.status_code == 200:
        # Save the generated image
        image = Image.open(io.BytesIO(response.content))
        image.save("generated_character.png")
        print("Image generated successfully!")
        return image
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage
generate_character(
    prompt="a beautiful elf warrior with golden armor",
    reference_image_path="reference.png"  # Optional
)
```

### Using Generated Images as References in Python

Here's how to generate multiple images using previous generations as references:

```python
def generate_character_sequence(
    base_prompt: str,
    variation_prompts: list[str],
    strength: float = 0.8
):
    # Generate initial character
    first_image = generate_character(prompt=base_prompt)
    if not first_image:
        return
    
    # Save first image
    first_image.save("character_1.png")
    
    # Generate variations using the first image as reference
    for i, prompt in enumerate(variation_prompts, 2):
        variation = generate_character(
            prompt=prompt,
            reference_image_path="character_1.png",
            strength=strength
        )
        if variation:
            variation.save(f"character_{i}.png")

# Example: Generate variations of a character
generate_character_sequence(
    base_prompt="a warrior elf with golden armor",
    variation_prompts=[
        "same warrior elf but in battle pose",
        "same warrior elf but with raised sword"
    ],
    strength=0.8
)
```

## 📚 Complete Python Example

Here's a complete example of the character creation workflow using Python:

```python
import requests
from PIL import Image
import time
from pathlib import Path
import json

class CharacterGenerator:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def create_character(self, prompt: str, existing_image_path: str = None) -> dict:
        """Create a new character"""
        url = f"{self.base_url}/characters/create_initial"
        
        if existing_image_path:
            files = {
                "existing_image": open(existing_image_path, "rb")
            }
            data = {"prompt": prompt}
            response = requests.post(url, files=files, data=data)
        else:
            response = requests.post(url, json={"prompt": prompt})
            
        response.raise_for_status()
        return response.json()

    def regenerate_base(self, character_id: str, prompt: str) -> dict:
        """Regenerate the base image"""
        url = f"{self.base_url}/characters/create_initial"
        data = {
            "prompt": prompt,
            "regenerate": True,
            "character_id": character_id
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def approve_base(self, character_id: str) -> dict:
        """Approve the current base image"""
        url = f"{self.base_url}/characters/{character_id}/approve"
        response = requests.post(url)
        response.raise_for_status()
        return response.json()

    def generate_training_data(
        self,
        character_id: str,
        num_variations: int = 10,
        custom_prompts: list = None
    ) -> dict:
        """Generate training data variations"""
        url = f"{self.base_url}/characters/{character_id}/generate_training"
        data = {
            "num_variations": num_variations,
            "custom_prompts": custom_prompts
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def start_training(self, character_id: str) -> dict:
        """Start LoRA training"""
        url = f"{self.base_url}/characters/{character_id}/train"
        response = requests.post(url)
        response.raise_for_status()
        return response.json()

    def get_training_status(self, character_id: str) -> dict:
        """Get current training status"""
        url = f"{self.base_url}/characters/{character_id}/training_status"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def wait_for_training(self, character_id: str, check_interval: int = 30) -> dict:
        """Wait for training to complete"""
        while True:
            status = self.get_training_status(character_id)
            state = status["training_status"]["state"]
            
            if state == "ready":
                return status
            elif state == "error":
                raise Exception(f"Training failed: {status['training_status'].get('error')}")
                
            print(f"Training progress: {status['training_status'].get('progress', 0):.1f}%")
            time.sleep(check_interval)

    def generate_scene(
        self,
        character_id: str,
        prompt: str,
        output_path: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> str:
        """Generate a new scene with the character"""
        url = f"{self.base_url}/characters/{character_id}/generate_scene"
        data = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        # Save the image
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        return output_path

# Example usage
def create_character_workflow():
    generator = CharacterGenerator()
    
    # 1. Create initial character
    character = generator.create_character(
        prompt="a noble elven warrior with golden armor"
    )
    character_id = character["character"]["id"]
    
    # 2. Regenerate until satisfied
    while input("Satisfied with the base image? (y/n): ").lower() != 'y':
        character = generator.regenerate_base(
            character_id,
            prompt="a noble elven warrior with golden armor"
        )
    
    # 3. Approve base image
    generator.approve_base(character_id)
    
    # 4. Generate training data
    training_result = generator.generate_training_data(
        character_id,
        num_variations=10,
        custom_prompts=[
            "character in battle pose",
            "character casting a spell",
            "character riding a horse",
            "character in formal attire",
            "character in stealth mode"
        ]
    )
    
    # 5. Start training
    generator.start_training(character_id)
    
    # 6. Wait for training to complete
    generator.wait_for_training(character_id)
    
    # 7. Generate scenes
    scenes = [
        "the character exploring an ancient temple",
        "the character in an epic battle",
        "the character in a peaceful elven village"
    ]
    
    for i, scene in enumerate(scenes):
        generator.generate_scene(
            character_id,
            prompt=scene,
            output_path=f"scene_{i+1}.png"
        )

if __name__ == "__main__":
    create_character_workflow()

### Advanced Training Features

#### Training Checkpoints

The training process automatically saves checkpoints that can be used to resume training if interrupted:

```bash
# Resume from latest checkpoint
curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/train/resume"

# Resume from specific checkpoint
curl -X POST "http://localhost:8000/characters/YOUR_CHARACTER_ID/train/resume" \
     -H "Content-Type: application/json" \
     -d '{
         "checkpoint_path": "/path/to/checkpoint.pt"
     }'
```

#### Training Metrics

The training status endpoint provides detailed metrics:

```json
{
    "status": "success",
    "training_status": {
        "state": "training",
        "progress": 45.5,
        "training_loss": 0.234,
        "epoch_loss": 0.245,
        "current_epoch": 5,
        "training_start": "2024-03-14T10:30:00",
        "last_update": "2024-03-14T10:35:00"
    }
}
```

#### Memory Management

The training process includes automatic memory management:
- Mixed precision training (FP16)
- Gradient clipping
- Automatic OOM recovery
- GPU memory cleanup

## 📋 API Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| prompt | string | Yes | - | Text description of the desired image |
| reference_image | file | No | None | Reference image for style guidance |
| num_inference_steps | int | No | 30 | Number of denoising steps |
| guidance_scale | float | No | 7.5 | How closely to follow the prompt |
| strength | float | No | 0.8 | How much to preserve from reference image |

## 🛠 Development

### Dependencies

Key Python packages required:
- torch >= 2.0.0
- diffusers >= 0.21.0
- transformers >= 4.31.0
- accelerate >= 0.21.0
- peft >= 0.5.0
- fastapi >= 0.100.0
- uvicorn >= 0.23.0

See `requirements.txt` for the complete list of dependencies.

### Project Structure

```
character-generator/
├── src/
│   ├── api.py            # FastAPI endpoints
│   ├── model_handler.py  # Stable Diffusion handler
│   ├── config.py         # Configuration classes
│   ├── character_manager.py # Character and LoRA management
│   └── main.py          # Application entry point
├── storage/
│   ├── base_characters/ # Character base images
│   ├── lora_models/    # Trained LoRA weights
│   └── outputs/        # Generated images
├── models/             # AI model storage
├── scripts/           # Utility scripts
├── docker-compose.yml # Docker configuration
├── Dockerfile        # Container definition
└── requirements.txt  # Python dependencies
```

### Running in Development Mode

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
python -m src.main
```

## ⚠️ Notes

- The service requires a CUDA-capable NVIDIA GPU
- First request might be slower due to model loading
- Make sure to have enough disk space for the AI models (~20GB)
- The API will return a PNG image file directly in the response
- All errors will return appropriate HTTP status codes with error messages

## 📝 Dependencies

Key dependencies (see requirements.txt for full list):
- torch >= 2.0.0
- diffusers >= 0.21.0
- transformers >= 4.31.0
- fastapi >= 0.100.0
- Pillow >= 10.0.0
- python-multipart >= 0.0.6