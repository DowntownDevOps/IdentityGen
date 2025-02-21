#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Change to src directory
os.chdir(str(src_path))

# Import and run the app
from api import app
from config import APIConfig

if __name__ == "__main__":
    import uvicorn
    
    config = APIConfig()
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=True
    ) 