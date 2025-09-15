"""
Simplified Configuration for Local-only Setup
No MongoDB, Redis, or complex dependencies required.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class SimpleSettings(BaseSettings):
    # App Configuration
    APP_NAME: str = "AI Class Notes Assistant (Simple)"
    APP_VERSION: str = "1.0.0-simple"
    DEBUG: bool = True
    
    # API Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # OpenAI Configuration (only required external dependency)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"  # More affordable than GPT-4
    
    # Local Storage Configuration
    DATA_DIR: Path = Path("data")
    UPLOAD_FOLDER: Path = Path("data/uploads")
    OUTPUT_FOLDER: Path = Path("data/generated")
    VECTOR_STORE_PATH: Path = Path("data/vectors")
    DOCUMENTS_DB_PATH: Path = Path("data/documents.json")
    
    # File Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB (reduced from 50MB)
    ALLOWED_EXTENSIONS: set = {"pdf", "txt", "docx"}  # Removed image processing
    
    # Text Processing
    CHUNK_SIZE: int = 500  # Reduced for simpler processing
    CHUNK_OVERLAP: int = 50
    
    # Embedding Configuration (using free/local models)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Lightweight sentence transformer
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("data/logs/app.log")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # Ignore extra fields instead of forbidding them
    )
    
    def __post_init__(self):
        """Create necessary directories"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Create a global settings instance
def get_simple_settings() -> SimpleSettings:
    settings = SimpleSettings()
    settings.__post_init__()
    return settings

settings = get_simple_settings()
