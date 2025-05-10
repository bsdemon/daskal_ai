import logging
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status
from src.api.routes import api_router
from src.core.config import dynamic_settings as settings
from src.core.dependencies import get_api_key
from src.db.config_db import config_db
from pathlib import Path
from log.custom_logging import CustomizeLogger

# logger = logging.getLogger('uvicorn.error')
logger = logging.getLogger(__name__)
config_path=Path(__file__).with_name("logging_config.json")
logger = CustomizeLogger.make_logger(config_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # TODO Add more to startup:
    config_db.initialize_default_settings()
    print("Starting up! Configuration:")
    # print(settings._base_settings.model_dump_json(indent=2))
    if not settings.base_settings.ENABLE_EMBEDDING:
        logger.warning("==== Embedding service is disabled. Please check your settings. ===")
    yield
    # TODO add more to Shutdown

app = FastAPI(
    title="RAG API",
    description="FastAPI RAG backend with ChromaDB",
    version="0.1.0",
    lifespan=lifespan,
)

# Add authentication middleware
app.include_router(api_router, prefix="/api", dependencies=[Depends(get_api_key)])

@app.get("/")
async def root():
    return {"message": "RAG API is running"}

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint for Kubernetes."""
    return {"status": "ok"} 

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
