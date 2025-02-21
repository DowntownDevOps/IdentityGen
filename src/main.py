import os
import sys
from pathlib import Path
import logging
import uvicorn

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(src_dir))

# Log the Python path for debugging
logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Contents of src directory: {list(src_dir.glob('*'))}")

from config import APIConfig

if __name__ == "__main__":
    logger.info("Starting server...")
    config = APIConfig()
    logger.info(f"Server config: {config}")
    
    uvicorn.run(
        "api:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=True,
        timeout_keep_alive=300
    ) 