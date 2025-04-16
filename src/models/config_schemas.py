from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class ConfigSetting(BaseModel):
    """Schema for a configuration setting."""

    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")
    value_type: str = Field(..., description="Configuration value type")
    description: Optional[str] = Field(
        None, description="Description of the configuration setting"
    )
    group_name: Optional[str] = Field(None, description="Group this setting belongs to")
    created_at: Optional[datetime] = Field(
        None, description="When this setting was created"
    )
    updated_at: Optional[datetime] = Field(
        None, description="When this setting was last updated"
    )


class ConfigSettingCreate(BaseModel):
    """Schema for creating a configuration setting."""

    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")
    value_type: Optional[str] = Field(None, description="Configuration value type")
    description: Optional[str] = Field(
        None, description="Description of the configuration setting"
    )
    group_name: Optional[str] = Field(None, description="Group this setting belongs to")


class ConfigSettingUpdate(BaseModel):
    """Schema for updating a configuration setting."""

    value: Any = Field(..., description="New configuration value")
    value_type: Optional[str] = Field(None, description="New configuration value type")
    description: Optional[str] = Field(None, description="New description")
    group_name: Optional[str] = Field(None, description="New group name")


class ConfigSettingResponse(BaseModel):
    """Schema for a configuration setting response."""

    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")
    value_type: str = Field(..., description="Configuration value type")
    description: Optional[str] = Field(
        None, description="Description of the configuration setting"
    )
    group_name: Optional[str] = Field(None, description="Group this setting belongs to")


class ConfigSettingsResponse(BaseModel):
    """Schema for multiple configuration settings."""

    settings: List[ConfigSettingResponse] = Field(
        ..., description="List of configuration settings"
    )


class ConfigGroupResponse(BaseModel):
    """Schema for configuration settings by group."""

    group_name: str = Field(..., description="Group name")
    settings: Dict[str, Any] = Field(..., description="Group settings")
