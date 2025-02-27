from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from character_manager import CharacterManager
from config import ModelConfig

app = FastAPI(title="Character Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize character manager
model_config = ModelConfig()
character_manager = CharacterManager(model_config)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Character Generator API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 