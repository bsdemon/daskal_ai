from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any

from src.models.config_schemas import (
    ConfigSettingCreate,
    ConfigSettingUpdate,
    ConfigSettingResponse,
    ConfigSettingsResponse,
    ConfigGroupResponse,
)
from src.models.settings_schemas import (
    AppSettings,
    AppSettingsUpdate,
)
from src.db.config_db import config_db
from src.core.dependencies import get_api_key

config_router = APIRouter(
    prefix="/config", tags=["config"], dependencies=[Depends(get_api_key)]
)


@config_router.get("/", response_model=ConfigSettingsResponse)
async def get_all_settings():
    """Get all configuration settings."""
    settings = config_db.get_all_settings()
    return ConfigSettingsResponse(
        settings=[
            ConfigSettingResponse(
                key=setting["key"],
                value=setting["value"],
                value_type=setting["value_type"],
                description=setting["description"],
                group_name=setting["group_name"],
            )
            for setting in settings
        ]
    )


@config_router.get("/groups", response_model=List[str])
async def get_setting_groups():
    """Get all configuration setting groups."""
    settings = config_db.get_all_settings()
    groups = set(setting["group_name"] for setting in settings if setting["group_name"])
    return sorted(list(groups))


@config_router.get("/group/{group_name}", response_model=ConfigGroupResponse)
async def get_group_settings(group_name: str):
    """Get all settings for a specific group."""
    settings = config_db.get_settings_by_group(group_name)
    if not settings:
        raise HTTPException(status_code=404, detail=f"Group '{group_name}' not found")

    return ConfigGroupResponse(group_name=group_name, settings=settings)


@config_router.get("/key/{key}", response_model=ConfigSettingResponse)
async def get_setting(key: str):
    """Get a specific configuration setting."""
    # First get all settings to get the full metadata
    all_settings = config_db.get_all_settings()
    setting = next((s for s in all_settings if s["key"] == key), None)

    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")

    return ConfigSettingResponse(
        key=setting["key"],
        value=setting["value"],
        value_type=setting["value_type"],
        description=setting["description"],
        group_name=setting["group_name"],
    )


@config_router.post("/", response_model=ConfigSettingResponse)
async def create_setting(setting: ConfigSettingCreate):
    """Create a new configuration setting."""
    # Check if setting already exists
    existing = config_db.get_setting(setting.key)
    if existing is not None:
        raise HTTPException(
            status_code=409, detail=f"Setting '{setting.key}' already exists"
        )

    # Create the setting
    config_db.set_setting(
        key=setting.key,
        value=setting.value,
        value_type=setting.value_type,
        description=setting.description,
        group_name=setting.group_name,
    )

    # Return the created setting
    return await get_setting(setting.key)


@config_router.put("/{key}", response_model=ConfigSettingResponse)
async def update_setting(key: str, setting_update: ConfigSettingUpdate):
    """Update an existing configuration setting."""
    # Check if setting exists
    existing = config_db.get_setting(key)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")

    # Update the setting
    config_db.set_setting(
        key=key,
        value=setting_update.value,
        value_type=setting_update.value_type,
        description=setting_update.description,
        group_name=setting_update.group_name,
    )

    # Return the updated setting
    return await get_setting(key)


@config_router.delete("/{key}")
async def delete_setting(key: str):
    """Delete a configuration setting."""
    deleted = config_db.delete_setting(key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")

    return {"message": f"Setting '{key}' deleted successfully"}


@config_router.post("/initialize")
async def initialize_settings():
    """Initialize the configuration database with default settings."""
    config_db.initialize_default_settings()
    return {"message": "Default settings initialized successfully"}


@config_router.get("/app", response_model=AppSettings)
async def get_app_settings():
    """Get all application settings."""
    # Retrieve all settings from the database
    all_settings = config_db.get_all_settings()
    settings_dict = {s["key"]: s["value"] for s in all_settings}
    
    # Convert database settings to camelCase keys for AppSettings
    return AppSettings(
        enable_embedding=settings_dict.get("ENABLE_EMBEDDING", True),
        enable_reranking=settings_dict.get("ENABLE_RERANKING", True),
        chunk_size=settings_dict.get("CHUNK_SIZE", 512),
        chunk_overlap=settings_dict.get("CHUNK_OVERLAP", 50),
        max_chunks=settings_dict.get("MAX_CHUNKS", 10),
        temperature=settings_dict.get("TEMPERATURE", 0.7),
        default_llm_provider=settings_dict.get("DEFAULT_LLM_PROVIDER", "anthropic"),
        default_embedding_provider=settings_dict.get("DEFAULT_EMBEDDING_PROVIDER", "voyage"),
        default_reranking_method=settings_dict.get("DEFAULT_RERANKING_METHOD", "bm25"),
    )


@config_router.patch("/app-settings", response_model=AppSettings)
async def update_app_settings(settings_update: AppSettingsUpdate):
    """Update application settings."""
    # First, get current settings
    current_settings = await get_app_settings()
    
    # Update settings in the database
    if settings_update.enable_embedding is not None:
        config_db.set_setting(
            key="ENABLE_EMBEDDING",
            value=settings_update.enable_embedding,
            value_type="bool",
            description="Enable embedding service",
            group_name="features",
        )
    
    
    if settings_update.enable_reranking is not None:
        config_db.set_setting(
            key="ENABLE_RERANKING",
            value=settings_update.enable_reranking,
            value_type="bool",
            description="Enable reranking service",
            group_name="features",
        )
    
    if settings_update.chunk_size is not None:
        config_db.set_setting(
            key="CHUNK_SIZE",
            value=settings_update.chunk_size,
            value_type="int",
            description="Size of text chunks for RAG",
            group_name="rag",
        )
    
    if settings_update.chunk_overlap is not None:
        config_db.set_setting(
            key="CHUNK_OVERLAP",
            value=settings_update.chunk_overlap,
            value_type="int",
            description="Overlap between text chunks",
            group_name="rag",
        )
    
    if settings_update.max_chunks is not None:
        config_db.set_setting(
            key="MAX_CHUNKS",
            value=settings_update.max_chunks,
            value_type="int",
            description="Maximum number of chunks to retrieve",
            group_name="rag",
        )
    
    if settings_update.temperature is not None:
        config_db.set_setting(
            key="TEMPERATURE",
            value=settings_update.temperature,
            value_type="float",
            description="Temperature for LLM generation",
            group_name="rag",
        )
    
    if settings_update.default_llm_provider is not None:
        config_db.set_setting(
            key="DEFAULT_LLM_PROVIDER",
            value=settings_update.default_llm_provider,
            value_type="str",
            description="Default LLM provider: anthropic, openai, or gemini",
            group_name="providers",
        )
    
    if settings_update.default_embedding_provider is not None:
        config_db.set_setting(
            key="DEFAULT_EMBEDDING_PROVIDER",
            value=settings_update.default_embedding_provider,
            value_type="str",
            description="Default embedding provider",
            group_name="providers",
        )
    
    if settings_update.default_reranking_method is not None:
        config_db.set_setting(
            key="DEFAULT_RERANKING_METHOD",
            value=settings_update.default_reranking_method,
            value_type="str",
            description="Default reranking method: bm25, cohere",
            group_name="providers",
        )
    
    # Return the updated settings
    return await get_app_settings()
