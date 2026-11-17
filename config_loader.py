import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
ENV_PATH = BASE_DIR / ".env"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class LocalQueueConfig(BaseModel):
    max_size: int = 5
    same_session_timeout_sec: float = 45.0
    grace_period_sec: float = 60.0
    local_only_timeout_sec: float = 120.0


class LocalCacheConfig(BaseModel):
    idle_clear_sec: int = 1800
    auto_clear_on_context_shift: bool = True


class LocalBackendConfig(BaseModel):
    url: str = "http://127.0.0.1:8080"
    model_name: str = "Qwen3.6-35B-A3B"
    max_context: int = 262144
    health_timeout_sec: float = 0.5
    queue: LocalQueueConfig = Field(default_factory=LocalQueueConfig)
    cache: LocalCacheConfig = Field(default_factory=LocalCacheConfig)


class ModelEntry(BaseModel):
    id: str
    tags: List[str] = Field(default_factory=list)
    tier: Optional[str] = None


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str
    enabled: bool = True
    use_proxy: bool = False
    timeout_sec: float = 60.0
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    models: List[ModelEntry] = Field(default_factory=list)
    only_free_models: bool = False


class AliasConfig(BaseModel):
    description: str = ""
    models: List[str] = Field(default_factory=list)  # Приоритет конкретных моделей (Model-First)
    provider_priority: List[str] = Field(default_factory=list)  # Приоритет провайдеров для одинаковых моделей
    priority: List[str] = Field(default_factory=list)  # Совместимость с Provider-First
    require_tools: bool = False
    require_vision: bool = False
    prefer_tags: List[str] = Field(default_factory=list)
    default_sampling: Dict[str, Any] = Field(default_factory=dict)


class ImageLocalConfig(BaseModel):
    enabled: bool = True
    max_side_px: int = 896
    output_format: str = "JPEG"
    quality: int = 85
    convert_transparent_to_bg: str = "#FFFFFF"
    grayscale: bool = False


class ImageCloudConfig(BaseModel):
    enabled: bool = True
    max_side_px: int = 2048
    quality: int = 90


class ImageProcessingConfig(BaseModel):
    enabled: bool = True
    local: ImageLocalConfig = Field(default_factory=ImageLocalConfig)
    cloud: ImageCloudConfig = Field(default_factory=ImageCloudConfig)


class RoutingConfig(BaseModel):
    sticky_sessions: bool = True
    session_ttl_sec: int = 7200
    key_cooldown_sec: float = 60.0
    discovery_refresh_hours: float = 6.0
    model_selection_mode: str = "user_first"  # "user_first" (сначала ручные, потом эвристика) или "user_only" (строго только ручные)
    provider_priority: List[str] = Field(
        default_factory=lambda: ["local", "cerebras", "groq", "sambanova", "openrouter"]
    )


class LLMClassifierConfig(BaseModel):
    enabled: bool = True
    provider: str = "cerebras"  # Провайдер для классификации (cerebras, groq, openrouter, local)
    model: str = "llama3.1-8b"
    batch_size: int = 25
    timeout_sec: float = 20.0


class ContextTiersConfig(BaseModel):
    huge: int = 131072
    large: int = 65536
    medium: int = 32768
    small: int = 8192


class DiscoveryConfig(BaseModel):
    enabled: bool = True
    refresh_hours: float = 6.0
    llm_classifier: LLMClassifierConfig = Field(default_factory=LLMClassifierConfig)
    context_tiers: ContextTiersConfig = Field(default_factory=ContextTiersConfig)


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    local_backend: LocalBackendConfig = Field(default_factory=LocalBackendConfig)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    aliases: Dict[str, AliasConfig] = Field(default_factory=dict)
    image_processing: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)


class Secrets(BaseModel):
    gateway_api_key: str = "gw-secret-key-12345"
    provider_keys: Dict[str, List[str]] = Field(default_factory=dict)
    outbound_proxy: Optional[str] = None


_CURRENT_CONFIG: Optional[AppConfig] = None
_CURRENT_SECRETS: Optional[Secrets] = None


def load_secrets() -> Secrets:
    env_data = dotenv_values(ENV_PATH)
    gateway_key = env_data.get("GATEWAY_API_KEY") or os.environ.get("GATEWAY_API_KEY", "gw-secret-key-12345")
    proxy = env_data.get("OUTBOUND_PROXY") or os.environ.get("OUTBOUND_PROXY", None)
    if proxy and not proxy.strip():
        proxy = None

    # Load all provider keys
    provider_keys: Dict[str, List[str]] = {}
    merged_env = {**os.environ, **{k: v for k, v in env_data.items() if v is not None}}
    for k, v in merged_env.items():
        if k.endswith("_API_KEYS") and v:
            keys = [key.strip() for key in v.split(",") if key.strip()]
            if keys:
                provider_keys[k] = keys

    return Secrets(
        gateway_api_key=gateway_key,
        provider_keys=provider_keys,
        outbound_proxy=proxy,
    )


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw)


def get_config() -> AppConfig:
    global _CURRENT_CONFIG
    if _CURRENT_CONFIG is None:
        _CURRENT_CONFIG = load_config()
    return _CURRENT_CONFIG


def get_secrets() -> Secrets:
    global _CURRENT_SECRETS
    if _CURRENT_SECRETS is None:
        _CURRENT_SECRETS = load_secrets()
    return _CURRENT_SECRETS


def reload_config() -> tuple[AppConfig, Secrets]:
    global _CURRENT_CONFIG, _CURRENT_SECRETS
    new_cfg = load_config()
    new_sec = load_secrets()
    _CURRENT_CONFIG = new_cfg
    _CURRENT_SECRETS = new_sec
    return _CURRENT_CONFIG, _CURRENT_SECRETS
