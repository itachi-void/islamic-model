# -*- coding: utf-8 -*-
import os

class Settings:
    # Model configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:3b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    
    # Vector store configuration
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/chroma")
    
    # Retrieval configurations
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "50"))

    # LLM API configuration (Groq / OpenAI compatible / Ollama)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")  # "auto", "groq", "openai", "ollama"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

settings = Settings()
