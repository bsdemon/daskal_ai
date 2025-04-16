import uvicorn
from fastapi import FastAPI, Depends
from src.api.routes import api_router
from src.core.config import dynamic_settings as settings
from src.core.dependencies import get_api_key
from src.db.config_db import config_db

app = FastAPI(
    title="RAG API",
    description="FastAPI RAG backend with ChromaDB",
    version="0.1.0",
)

# Add authentication middleware
app.include_router(api_router, prefix="/api", dependencies=[Depends(get_api_key)])


@app.on_event("startup")
async def startup_event():
    """Initialize the configuration database on startup."""
    config_db.initialize_default_settings()


@app.get("/")
async def root():
    return {"message": "RAG API is running"}


if __name__ == "__main__":
    print("Starting server...")
    print(f"Using settings: {settings}")
    print(f"API Key: {settings.API_KEY}")
    print(f"ChromaDB URL: {settings.CHROMA_URL}")
    print(f"ChromaDB Collection Name: {settings.CHROMA_COLLECTION_NAME}")
    print(f"ChromaDB Persist Directory: {settings.CHROMA_PERSIST_DIRECTORY}")
    print(f"API_PRESHARED_KEY: {settings.API_PRESHARED_KEY}")
    
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
