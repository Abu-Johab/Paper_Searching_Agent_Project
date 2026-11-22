"""
hsi_research_agent/config.py
Configuration module for the Autonomous Academic Research Discovery Agent.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Settings:
    """Application settings and runtime configuration."""
    
    BASE_DIR: Path = BASE_DIR
    
    # Ollama Local LLM Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    # Storage and Persistence
    DATA_DIR: Path = BASE_DIR / "data"
    EXPORTS_DIR: Path = BASE_DIR / "exports"
    LOGS_DIR: Path = BASE_DIR / "logs"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"
    DB_PATH: Path = DATA_DIR / os.getenv("DB_NAME", "research_agent.db")
    
    # API & Search Configuration
    CONTACT_EMAIL: str = os.getenv("CONTACT_EMAIL", "research_agent@example.com").strip()
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None
    IEEE_API_KEY: Optional[str] = os.getenv("IEEE_API_KEY", "").strip() or None
    UNPAYWALL_EMAIL: str = os.getenv("UNPAYWALL_EMAIL", CONTACT_EMAIL).strip()
    
    # Polite API Rate Limits (minimum interval seconds)
    DELAY_CROSSREF: float = 0.5
    DELAY_ARXIV: float = 3.0
    DELAY_SEMANTIC_SCHOLAR: float = 1.0
    DELAY_OPENALEX: float = 0.2
    DELAY_PUBMED: float = 0.34
    DELAY_IEEE: float = 0.5
    
    # Literature Search Defaults
    DEFAULT_START_YEAR: int = 2018
    DEFAULT_END_YEAR: Optional[int] = None
    DEFAULT_MAX_RESULTS_PER_QUERY: int = 50
    EXACT_QUERY_COUNT: int = 10
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensure all required runtime directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate settings singleton
settings = Settings()
settings.ensure_dirs()
