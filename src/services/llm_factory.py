from abc import ABC, abstractmethod
from typing import Optional
from anthropic import AsyncAnthropic
from src.core.config import dynamic_settings as settings


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from the LLM."""
        pass


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(self):
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        temp = temperature if temperature is not None else settings.TEMPERATURE

        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            system=(
                system_prompt if system_prompt else "You are a helpful AI assistant."
            ),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temp,
        )
        print(response)
        return response.content[0].text


class OpenAIClient(LLMClient):
    """Client for OpenAI's API."""

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        # Import here to avoid dependency if not used
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        temp = temperature if temperature is not None else settings.TEMPERATURE

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=temp,
            messages=[
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        if system_prompt
                        else "You are a helpful AI assistant."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content


class GeminiClient(LLMClient):
    """Client for Google's Gemini API."""

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set")
        # Import here to avoid dependency if not used
        import google.generativeai as genai

        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.client = genai

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        temp = temperature if temperature is not None else settings.TEMPERATURE

        # Combine system prompt with user prompt for Gemini
        combined_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        model = self.client.GenerativeModel("gemini-pro")
        response = model.generate_content(combined_prompt, temperature=temp)

        return response.text


class LLMFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(provider: str = None) -> LLMClient:
        """Create an LLM client based on the provider."""
        provider = provider or settings.DEFAULT_LLM_PROVIDER

        if provider == "anthropic":
            return AnthropicClient()
        elif provider == "openai":
            return OpenAIClient()
        elif provider == "gemini":
            return GeminiClient()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
