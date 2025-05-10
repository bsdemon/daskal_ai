from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from src.core.config import dynamic_settings as settings

API_KEY_NAME = "Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != settings.API_PRESHARED_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key_header
