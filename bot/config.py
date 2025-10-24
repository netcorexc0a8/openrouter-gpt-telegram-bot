"""
Module for loading and validating bot configuration.

This module provides classes and functions for:
- Defining configuration data structures
- Loading configuration from YAML files
- Validating configuration values
- Parsing comma-separated integer lists
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import os


@dataclass
class ModelParameters:
    """
    Class representing model parameters configuration.
    
    Attributes:
        type: The type of model
        model_name: The name of the model to use
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0)
        min_p: Minimum probability threshold
        presence_penalty: Penalty for existing tokens (-2.0 to 2.0)
        repetition_penalty: Penalty for repeated tokens
        top_a: Top-a sampling parameter
        top_k: Top-k sampling parameter
    """
    type: str = ""
    model_name: str = ""
    temperature: float = 1.0
    top_p: float = 0.7
    frequency_penalty: float = 0.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 0.0
    top_a: float = 0.0
    top_k: float = 0.0


@dataclass
class Config:
    """
    Main configuration class that mirrors the Go struct.
    
    Attributes:
        telegram_bot_token: Telegram bot token
        openai_api_key: OpenAI API key
        model: Model parameters configuration
        max_tokens: Maximum tokens for API requests
        bot_language: Language for the bot
        openai_base_url: Base URL for OpenAI API
        system_prompt: Default system prompt
        budget_period: Budget period ('daily', 'monthly', 'total')
        guest_budget: Budget limit for guests
        user_budget: Budget limit for users
        admin_chat_ids: List of admin chat IDs
        allowed_user_chat_ids: List of allowed user chat IDs
        allowed_group_ids: List of allowed group IDs
        max_history_size: Maximum number of messages in history
        max_history_time: Maximum time for history in minutes
        vision: Vision capability flag
        vision_prompt: Prompt for vision capability
        vision_details: Details for vision capability
        stats_min_role: Minimum role to view stats
        lang: Language code
        api_requests_per_minute: Maximum API requests per minute
        api_tokens_per_minute: Maximum tokens per minute
        api_tokens_per_day: Maximum tokens per day
        api_concurrent_requests: Maximum concurrent requests
    """
    telegram_bot_token: str = ""
    openai_api_key: str = ""
    model: ModelParameters = field(default_factory=ModelParameters)
    max_tokens: int = 2000
    bot_language: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    system_prompt: str = ""
    budget_period: str = "monthly"
    guest_budget: float = 0.0
    user_budget: float = 0.0
    admin_chat_ids: List[int] = field(default_factory=list)
    allowed_user_chat_ids: List[int] = field(default_factory=list)
    allowed_group_ids: List[int] = field(default_factory=list)
    max_history_size: int = 10
    max_history_time: int = 60
    vision: str = ""
    vision_prompt: str = ""
    vision_details: str = ""
    stats_min_role: str = ""
    lang: str = "en"
    api_requests_per_minute: int = 30
    api_tokens_per_minute: int = 50000
    api_tokens_per_day: int = 500000
    api_concurrent_requests: int = 3
    # GCRA rate limiting configuration (for advanced rate limiting)
    gcra_requests_limit: Dict[str, Any] = field(default_factory=dict)  # {model_name: {"limit": int, "window": int, "burst": int}}
    gcra_tokens_limit: Dict[str, Any] = field(default_factory=dict)    # {model_name: {"limit": int, "window": int, "burst": int}}


def parse_int_list(value: str) -> List[int]:
    """
    Parse a comma-separated string of integers into a list.
    
    Args:
        value: Comma-separated string of integers
        
    Returns:
        List of integers parsed from the string
    """
    if not value:
        return []
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        import logging
        logging.warning(f"Warning: Invalid integer values in list: {value}")
        return []


def validate_config(config: Config) -> None:
    """
    Validate the configuration values.
    
    Args:
        config: The configuration object to validate
        
    Raises:
        ValueError: If any configuration value is invalid
    """
    errors = []
    
    # Validate required fields
    if not config.telegram_bot_token:
        errors.append("telegram_bot_token is required")
    
    if not config.openai_api_key:
        errors.append("openai_api_key is required")
    
    if not config.model.model_name:
        errors.append("model_name is required")
    
    # Validate budget period
    valid_budget_periods = ["daily", "monthly", "total"]
    if config.budget_period not in valid_budget_periods:
        errors.append(f"budget_period must be one of {valid_budget_periods}, got '{config.budget_period}'")
    
    # Validate numeric values
    if config.max_tokens <= 0:
        errors.append("max_tokens must be greater than 0")
    
    if config.model.temperature < 0 or config.model.temperature > 2:
        errors.append("model.temperature must be between 0 and 2")
    
    if config.model.top_p < 0 or config.model.top_p > 1:
        errors.append("model.top_p must be between 0 and 1")
    
    if config.model.frequency_penalty < -2 or config.model.frequency_penalty > 2:
        errors.append("model.frequency_penalty must be between -2 and 2")
    
    if config.model.presence_penalty < -2 or config.model.presence_penalty > 2:
        errors.append("model.presence_penalty must be between -2 and 2")
    
    if config.user_budget < 0:
        errors.append("user_budget must be non-negative")
    
    if config.guest_budget < 0:
        errors.append("guest_budget must be non-negative")
    
    if config.max_history_size <= 0:
        errors.append("max_history_size must be greater than 0")
    
    if config.max_history_time <= 0:
        errors.append("max_history_time must be greater than 0")
    
    # Validate language
    valid_languages = ["en", "ru"]  # Add more as needed
    if config.lang not in valid_languages:
        errors.append(f"lang must be one of {valid_languages}, got '{config.lang}'")
    
    if errors:
        raise ValueError("Configuration validation failed: " + "; ".join(errors))


def load_config(path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Config object with validated configuration values
        
    Raises:
        ValueError: If configuration validation fails
    """
    # Load environment variables
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    openai_api_key = os.getenv("API_KEY", "")
    model_from_env = os.getenv("MODEL", "")
    
    # Check if config file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Read and parse YAML configuration
    with open(path, 'r', encoding='utf-8') as file:
        yaml_config = yaml.safe_load(file)
    
    # Create model parameters from YAML config, with .env MODEL taking precedence
    model_params = ModelParameters(
        type=yaml_config.get("type", ""),
        model_name=model_from_env if model_from_env else yaml_config.get("model", ""),
        temperature=yaml_config.get("temperature", 1.0),
        top_p=yaml_config.get("top_p", 0.7),
        frequency_penalty=yaml_config.get("frequency_penalty", 0.0),
        min_p=yaml_config.get("min_p", 0.0),
        presence_penalty=yaml_config.get("presence_penalty", 0.0),
        repetition_penalty=yaml_config.get("repetition_penalty", 0.0),
        top_a=yaml_config.get("top_a", 0.0),
        top_k=yaml_config.get("top_k", 0.0)
    )
    
    # Create main config object
    config = Config(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        model=model_params,
        max_tokens=yaml_config.get("max_tokens", 2000),
        bot_language=yaml_config.get("bot_language", ""),
        openai_base_url=yaml_config.get("base_url", "https://api.openai.com/v1"),
        system_prompt=yaml_config.get("assistant_prompt", ""),
        budget_period=yaml_config.get("budget_period", "monthly"),
        guest_budget=yaml_config.get("guest_budget", 0.0),
        user_budget=yaml_config.get("user_budget", 0.0),
        admin_chat_ids=parse_int_list(yaml_config.get("admin_ids", "")),
        allowed_user_chat_ids=parse_int_list(yaml_config.get("allowed_user_ids", "")),
        allowed_group_ids=parse_int_list(yaml_config.get("allowed_group_ids", "")),
        max_history_size=yaml_config.get("max_history_size", 10),
        max_history_time=yaml_config.get("max_history_time", 60),
        vision=yaml_config.get("vision", ""),
        vision_prompt=yaml_config.get("vision_prompt", ""),
        vision_details=yaml_config.get("vision_detail", ""),
        stats_min_role=yaml_config.get("stats_min_role", ""),
        lang=yaml_config.get("lang", "en"),
        api_requests_per_minute=yaml_config.get("api_requests_per_minute", 60),
        api_tokens_per_minute=yaml_config.get("api_tokens_per_minute", 100000),
        api_tokens_per_day=yaml_config.get("api_tokens_per_day", 1000000),
        api_concurrent_requests=yaml_config.get("api_concurrent_requests", 5)
    )
    
    # Load GCRA rate limiting configurations if present
    config.gcra_requests_limit = yaml_config.get("gcra_requests_limit", {})
    config.gcra_tokens_limit = yaml_config.get("gcra_tokens_limit", {})
    
    # Validate the configuration
    validate_config(config)
    
    # Add language instruction to system prompt
    from bot.localization import Localization
    localization = Localization()
    language = localization.get("language", config.lang)
    # Sanitize the language string to prevent injection
    import re
    safe_language = re.sub(r'[^a-zA-Z\s]', '', str(language))[:50]  # Limit to 50 alphanumeric characters
    config.system_prompt = f"Always answer in {safe_language} language." + config.system_prompt
    
    return config