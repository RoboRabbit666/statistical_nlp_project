"""Configuration management for the NLP project."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml


@dataclass
class Config:
    """Configuration class for NLP project settings."""
    
    # Model configurations
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    anthropic_model: str = "claude-3-haiku-20240307"
    max_tokens: int = 4096
    temperature: float = 0.0
    
    # Data configurations
    batch_size: int = 128
    max_sequence_length: int = 512
    num_retrieved_sentences: int = 5
    
    # API configurations
    anthropic_api_key: Optional[str] = None
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            sentence_model=os.getenv("SENTENCE_MODEL", cls.sentence_model),
            device=os.getenv("DEVICE", cls.device),
            log_level=os.getenv("LOG_LEVEL", cls.log_level)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "sentence_model": self.sentence_model,
            "anthropic_model": self.anthropic_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "num_retrieved_sentences": self.num_retrieved_sentences,
            "device": self.device,
            "log_level": self.log_level,
            "log_format": self.log_format
        }