from abc import ABC, abstractmethod
from typing import List
from src.core.config import dynamic_settings as settings


class EmbeddingClient(ABC):
    """Abstract base class for embedding clients."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the provided texts."""
        pass


class VoyageAIClient(EmbeddingClient):
    """Client for Voyage AI's embedding API."""

    def __init__(self):
        if not settings.VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY is not set")
        # Import here to avoid dependency if not used
        from voyageai import get_embeddings

        self.get_embeddings = get_embeddings
        self.api_key = settings.VOYAGE_API_KEY

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Voyage AI."""
        embeddings = self.get_embeddings(
            texts, model="voyage-large-2", api_key=self.api_key
        )
        return embeddings


class OpenAIEmbeddingClient(EmbeddingClient):
    """Client for OpenAI's embedding API."""

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        # Import here to avoid dependency if not used
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        # Process in batches to avoid context length issues
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.embeddings.create(
                model="text-embedding-3-large", input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class EmbeddingFactory:
    """Factory for creating embedding clients."""

    @staticmethod
    def create_client(provider: str = None) -> EmbeddingClient:
        """Create an embedding client based on the provider."""
        provider = provider or settings.DEFAULT_EMBEDDING_PROVIDER

        if provider == "voyage":
            return VoyageAIClient()
        elif provider == "openai":
            return OpenAIEmbeddingClient()
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
