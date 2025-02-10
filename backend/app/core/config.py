from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field, AnyHttpUrl


class Settings(BaseSettings):
    PROJECT_NAME: str = "Janus Pro API"
    DESCRIPTION: str = "Professional API for Janus Pro 1B Image Generation"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    SERVE_STATIC: bool = Field(False, env="SERVE_STATIC")
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    
    MODEL_NAME: str = "deepseek-ai/Janus-Pro-1B"
    TORCH_DTYPE: str = "bfloat16"
    DEVICE_MAP: str = "auto"
    
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost",
        "http://localhost:8080",
    ]
    
    DEFAULT_GEN_PARAMS: dict = {
        "temperature": 1.0,
        "parallel_size": 4,
        "cfg_weight": 5.0,
        "image_token_num_per_image": 576,
        "img_size": 384,
        "patch_size": 16,
    }
    
    MAX_CONCURRENT_REQUESTS: int = 4
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()