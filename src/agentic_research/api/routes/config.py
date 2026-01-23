"""
Configuration API routes for runtime settings management.

Allows the frontend to get/set configuration options like
LLM provider, API keys, and model selection.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/config", tags=["config"])


class ConfigUpdate(BaseModel):
    """Configuration update request."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None
    langextract_provider: Optional[str] = None
    langextract_model: Optional[str] = None


class ConfigResponse(BaseModel):
    """Current configuration (without sensitive data)."""
    llm_provider: str
    llm_model: str
    embedding_model: str
    ollama_base_url: str
    ollama_model: str
    langextract_provider: str
    langextract_model: str
    has_openai_key: bool
    has_anthropic_key: bool
    has_gemini_key: bool


@router.get("", response_model=ConfigResponse)
async def get_config():
    """Get current configuration (masks sensitive data)."""
    return ConfigResponse(
        llm_provider=os.environ.get("PRAVAL_DEFAULT_PROVIDER", "openai"),
        llm_model=os.environ.get("PRAVAL_DEFAULT_MODEL", "gpt-4o-mini"),
        embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        langextract_provider=os.environ.get("LANGEXTRACT_PROVIDER", "gemini"),
        langextract_model=os.environ.get("LANGEXTRACT_MODEL", "gemini-2.5-flash"),
        has_openai_key=bool(os.environ.get("OPENAI_API_KEY")),
        has_anthropic_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
        has_gemini_key=bool(os.environ.get("GEMINI_API_KEY")),
    )


@router.post("")
async def update_config(config: ConfigUpdate):
    """
    Update runtime configuration.

    Note: This updates environment variables for the current session.
    For persistent changes, use the desktop app settings or .env file.
    """
    valid_llm_providers = ["openai", "anthropic", "ollama"]
    valid_langextract_providers = ["gemini", "openai", "ollama"]

    if config.llm_provider and config.llm_provider not in valid_llm_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM provider. Must be one of: {valid_llm_providers}"
        )

    if config.langextract_provider and config.langextract_provider not in valid_langextract_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LangExtract provider. Must be one of: {valid_langextract_providers}"
        )

    # Update environment variables
    if config.openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key

    if config.anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key

    if config.gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = config.gemini_api_key

    if config.llm_provider:
        os.environ["PRAVAL_DEFAULT_PROVIDER"] = config.llm_provider

    if config.llm_model:
        os.environ["PRAVAL_DEFAULT_MODEL"] = config.llm_model

    if config.embedding_model:
        os.environ["OPENAI_EMBEDDING_MODEL"] = config.embedding_model

    if config.ollama_base_url:
        os.environ["OLLAMA_BASE_URL"] = config.ollama_base_url

    if config.ollama_model:
        os.environ["OLLAMA_MODEL"] = config.ollama_model

    if config.langextract_provider:
        os.environ["LANGEXTRACT_PROVIDER"] = config.langextract_provider

    if config.langextract_model:
        os.environ["LANGEXTRACT_MODEL"] = config.langextract_model

    return {"status": "ok", "message": "Configuration updated for current session"}


@router.get("/providers")
async def get_available_providers():
    """Get list of available LLM providers and their models."""
    return {
        "llm_providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "requires_key": True,
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                "requires_key": True,
            },
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "models": ["llama3.2", "llama3.1", "mistral", "mixtral", "codellama", "phi3", "gemma2:9b"],
                "requires_key": False,
            },
        ],
        "langextract_providers": [
            {
                "id": "gemini",
                "name": "Google Gemini",
                "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
                "requires_key": True,
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini"],
                "requires_key": True,
            },
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "models": ["llava", "bakllava", "llama3.2-vision"],
                "requires_key": False,
            },
        ],
        "embedding_models": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
    }
