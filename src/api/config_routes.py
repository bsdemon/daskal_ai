from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional

from src.models.config_schemas import (
    ConfigSetting,
    ConfigSettingCreate,
    ConfigSettingUpdate,
    ConfigSettingResponse,
    ConfigSettingsResponse,
    ConfigGroupResponse
)
from src.db.config_db import config_db
from src.core.dependencies import get_api_key

config_router = APIRouter(prefix="/config", tags=["config"], dependencies=[Depends(get_api_key)])

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
                group_name=setting["group_name"]
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
    
    return ConfigGroupResponse(
        group_name=group_name,
        settings=settings
    )

@config_router.get("/{key}", response_model=ConfigSettingResponse)
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
        group_name=setting["group_name"]
    )

@config_router.post("/", response_model=ConfigSettingResponse)
async def create_setting(setting: ConfigSettingCreate):
    """Create a new configuration setting."""
    # Check if setting already exists
    existing = config_db.get_setting(setting.key)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Setting '{setting.key}' already exists"
        )
    
    # Create the setting
    config_db.set_setting(
        key=setting.key,
        value=setting.value,
        value_type=setting.value_type,
        description=setting.description,
        group_name=setting.group_name
    )
    
    # Return the created setting
    return await get_setting(setting.key)

@config_router.put("/{key}", response_model=ConfigSettingResponse)
async def update_setting(key: str, setting_update: ConfigSettingUpdate):
    """Update an existing configuration setting."""
    # Check if setting exists
    existing = config_db.get_setting(key)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=f"Setting '{key}' not found"
        )
    
    # Update the setting
    config_db.set_setting(
        key=key,
        value=setting_update.value,
        value_type=setting_update.value_type,
        description=setting_update.description,
        group_name=setting_update.group_name
    )
    
    # Return the updated setting
    return await get_setting(key)

@config_router.delete("/{key}")
async def delete_setting(key: str):
    """Delete a configuration setting."""
    deleted = config_db.delete_setting(key)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Setting '{key}' not found"
        )
    
    return {"message": f"Setting '{key}' deleted successfully"}

@config_router.post("/initialize")
async def initialize_settings():
    """Initialize the configuration database with default settings."""
    config_db.initialize_default_settings()
    return {"message": "Default settings initialized successfully"}