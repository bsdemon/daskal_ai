from typing import Optional
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import threading


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Configuration
    API_PRESHARED_KEY: str = Field(
        ..., description="Preshared API key for authentication"
    )

    # AI Service Providers
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="Anthropic API key")
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    GEMINI_API_KEY: Optional[str] = Field(None, description="Google Gemini API key")

    # Embedding Service
    VOYAGE_API_KEY: Optional[str] = Field(
        None, description="Voyage AI API key for embeddings"
    )

    # Reranking Service
    COHERE_API_KEY: Optional[str] = Field(
        None, description="Cohere API key for reranking"
    )

    # Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = Field(
        "./chroma_db", description="ChromaDB persistence directory"
    )

    # Feature Flags
    ENABLE_EMBEDDING: bool = Field(True, description="Enable embedding service")
    ENABLE_RERANKING: bool = Field(True, description="Enable reranking service")

    # RAG Configuration
    CHUNK_SIZE: int = Field(512, description="Size of text chunks for RAG")
    CHUNK_OVERLAP: int = Field(50, description="Overlap between text chunks")
    MAX_CHUNKS: int = Field(10, description="Maximum number of chunks to retrieve")
    TEMPERATURE: float = Field(0.7, description="Temperature for LLM generation")

    # Default service configuration
    DEFAULT_LLM_PROVIDER: str = Field(
        "anthropic", description="Default LLM provider: anthropic, openai, or gemini"
    )
    DEFAULT_EMBEDDING_PROVIDER: str = Field(
        "voyage", description="Default embedding provider"
    )
    DEFAULT_RERANKING_METHOD: str = Field(
        "bm25", description="Default reranking method: bm25, cohere"
    )
    SYSTEM_PROMPT: str = Field(
        "You are a helpful AI assistant. Answer the user's question based on the provided context. "
        "If the answer is not found in the context, say so clearly and provide a general response. "
        "Always cite your sources using the numbers in brackets [1], [2], etc.",
        description="System prompt for LLM"
    )

settings = None
try:
    settings = Settings()
except ValidationError as e:
    print(f"Error loading settings: {e}")
    import sys
    sys.exit(0)  # Use sys.exit instead of exit()
except Exception as e:
    print(f"Unexpected error loading settings: {e}")


# Create a dynamic settings class that combines environment variables with database settings
class DynamicSettings:
    """Dynamic settings that combine environment variables with database settings."""

    def __init__(self, base_settings: Settings):
        self.base_settings = base_settings
        self._db = None  # Will be initialized lazily
        self._lock = threading.Lock()

    def _get_db(self):
        """Lazy-load the database connection."""
        if self._db is None:
            with self._lock:
                if self._db is None:
                    # Import here to avoid circular imports
                    from src.db.config_db import config_db

                    self._db = config_db
        return self._db

    def __getattr__(self, name):
        """Get a setting value, with database values taking precedence."""
        # Check if the attribute exists in the base settings
        if hasattr(self.base_settings, name):
            # Try to get from database first
            try:
                db = self._get_db()
                db_value = db.get_setting(name)
                if db_value is not None:
                    return db_value
            except Exception:
                # If database access fails, fall back to base settings
                pass

            # Fall back to base settings
            return getattr(self.base_settings, name)

        # If not found in base settings, raise AttributeError
        raise AttributeError(f"'DynamicSettings' object has no attribute '{name}'")


# Replace the static settings with the dynamic version
dynamic_settings = DynamicSettings(settings)
