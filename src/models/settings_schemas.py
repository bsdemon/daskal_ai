from typing import Optional
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    """Schema for application settings."""

    # Feature Flags
    enable_embedding: bool = Field(
        True, description="Enable embedding service"
    )
    enable_reranking: bool = Field(
        True, description="Enable reranking service"
    )

    # RAG Configuration
    chunk_size: int = Field(
        512, description="Size of text chunks for RAG", ge=32, le=2048
    )
    chunk_overlap: int = Field(
        50, description="Overlap between text chunks", ge=0, le=512
    )
    max_chunks: int = Field(
        10, description="Maximum number of chunks to retrieve", ge=1, le=100
    )
    temperature: float = Field(
        0.7, description="Temperature for LLM generation", ge=0.0, le=1.0
    )

    # Default Providers
    default_llm_provider: str = Field(
        "anthropic", description="Default LLM provider: anthropic, openai, or gemini"
    )
    default_embedding_provider: str = Field(
        "voyage", description="Default embedding provider"
    )
    default_reranking_method: str = Field(
        "bm25", description="Default reranking method: bm25, cohere"
    )


class AppSettingsUpdate(BaseModel):
    """Schema for updating application settings."""

    # Feature Flags
    enable_embedding: Optional[bool] = Field(
        None, description="Enable embedding service"
    )
    enable_reranking: Optional[bool] = Field(
        None, description="Enable reranking service"
    )

    # RAG Configuration
    chunk_size: Optional[int] = Field(
        None, description="Size of text chunks for RAG", ge=32, le=2048
    )
    chunk_overlap: Optional[int] = Field(
        None, description="Overlap between text chunks", ge=0, le=512
    )
    max_chunks: Optional[int] = Field(
        None, description="Maximum number of chunks to retrieve", ge=1, le=100
    )
    temperature: Optional[float] = Field(
        None, description="Temperature for LLM generation", ge=0.0, le=1.0
    )

    # Default Providers
    default_llm_provider: Optional[str] = Field(
        None, description="Default LLM provider: anthropic, openai, or gemini"
    )
    default_embedding_provider: Optional[str] = Field(
        None, description="Default embedding provider"
    )
    default_reranking_method: Optional[str] = Field(
        None, description="Default reranking method: bm25, cohere"
    )