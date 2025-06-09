# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration settings for the multi-agent system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google AI Studio / Gemini Configuration
    google_api_key: str
    google_project_id: str
    google_location: str = "us-central1"
    
    # Supabase Configuration
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    
    # WorkOS Authentication (optional)
    workos_api_key: Optional[str] = None
    workos_client_id: Optional[str] = None
    
    # Agent Configuration
    root_agent_model: str = "gemini-1.5-pro-002"
    agent_temperature: float = 0.1
    max_tokens: int = 8192
    
    # MCP Server Configuration
    mcp_server_port: int = 3000
    mcp_server_host: str = "localhost"
    
    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Production Configuration
    environment: str = "development"
    redis_url: Optional[str] = None
    database_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings() 