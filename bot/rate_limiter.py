"""
Module implementing GCRA (Generic Cell Rate Algorithm) for rate limiting.

This module provides classes and functions for:
- Implementing the Generic Cell Rate Algorithm (GCRA) for rate limiting
- Supporting per-user and per-model rate limiting
- Working without background processes
- Providing O(1) complexity for rate limit checks
- Allowing burst limits
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class LimitType(Enum):
    REQUESTS = "requests"
    TOKENS = "tokens"


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.
    
    Attributes:
        limit: Maximum number of units allowed
        window: Time window in seconds
        burst: Maximum burst size allowed
    """
    limit: int
    window: int
    burst: int = 0
    
    def __post_init__(self):
        if self.burst < 0:
            self.burst = 0
        if self.limit <= 0:
            raise ValueError("Limit must be greater than 0")
        if self.window <= 0:
            raise ValueError("Window must be greater than 0")


class GCRA:
    """
    Generic Cell Rate Algorithm implementation for rate limiting.
    
    This implementation provides O(1) complexity for rate limit checks without requiring
    background processes. It supports burst limits and can be configured individually
    for different users and models.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the GCRA rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        # tau (time constant) = window_size / limit
        self.tau = self.config.window / self.config.limit
        # Burst allowance: additional time allowed for bursts
        self.emission_interval = self.tau
        # The maximum delay we allow in the system (burst capacity)
        self.burst_delay = self.config.burst * self.tau if self.config.burst > 0 else 0
        # Initialize the time of the next allowed request to now
        self.tat = time.time() # Theoretical Arrival Time
    
    def allow(self, count: int = 1) -> tuple[bool, float]:
        """
        Check if a request is allowed under the rate limit.
        
        Args:
            count: Number of units to check (e.g., number of tokens or requests)
            
        Returns:
            A tuple of (allowed: bool, delay: float) where delay is the time to wait
            until the request would be allowed
        """
        current_time = time.time()
        
        # Calculate the required arrival time based on the emission interval
        # This is when the request should be allowed to arrive
        required_tat = self.tat + (count * self.emission_interval)
        
        # The time we need to wait is the difference between required arrival time
        # and current time
        delay_needed = required_tat - current_time
        
        # Check if we're within burst limits
        if delay_needed <= (self.burst_delay + 0.001):  # Small epsilon for floating point comparison
            # Update the theoretical arrival time
            self.tat = max(current_time, required_tat)
            return True, 0.0
        else:
            # The request exceeds the rate limit
            return False, delay_needed


class RateLimiter:
    """
    Manages multiple GCRA instances for different users and models.
    
    This class provides a central point for managing rate limits across different
    users and models, with individual configurations for each.
    """
    
    def __init__(self):
        """
        Initialize the rate limiter.
        """
        # Dictionary to store GCRA instances: {limit_type: {key: GCRA}}
        self.limiters: Dict[LimitType, Dict[str, GCRA]] = {
            LimitType.REQUESTS: {},
            LimitType.TOKENS: {}
        }
    
    def get_key(self, user_id: str, model: str, limit_type: LimitType) -> str:
        """
        Generate a unique key for a user-model combination.
        
        Args:
            user_id: The user ID
            model: The model name
            limit_type: The type of limit (requests or tokens)
            
        Returns:
            A unique key for this combination
        """
        return f"{user_id}:{model}:{limit_type.value}"
    
    def set_config(self, user_id: str, model: str, limit_type: LimitType, config: RateLimitConfig):
        """
        Set the rate limit configuration for a user-model combination.
        
        Args:
            user_id: The user ID
            model: The model name
            limit_type: The type of limit (requests or tokens)
            config: The rate limit configuration
        """
        key = self.get_key(user_id, model, limit_type)
        self.limiters[limit_type][key] = GCRA(config)
    
    def allow_request(self, user_id: str, model: str, count: int = 1) -> tuple[bool, float]:
        """
        Check if a request is allowed based on request rate limits.
        
        Args:
            user_id: The user ID
            model: The model name
            count: Number of requests to check
            
        Returns:
            A tuple of (allowed: bool, delay: float) where delay is the time to wait
        """
        key = self.get_key(user_id, model, LimitType.REQUESTS)
        
        if key not in self.limiters[LimitType.REQUESTS]:
            # If no specific config is set, allow the request (default behavior)
            return True, 0.0
        
        return self.limiters[LimitType.REQUESTS][key].allow(count)
    
    def allow_tokens(self, user_id: str, model: str, count: int = 1) -> tuple[bool, float]:
        """
        Check if a token usage is allowed based on token rate limits.
        
        Args:
            user_id: The user ID
            model: The model name
            count: Number of tokens to check
            
        Returns:
            A tuple of (allowed: bool, delay: float) where delay is the time to wait
        """
        key = self.get_key(user_id, model, LimitType.TOKENS)
        
        if key not in self.limiters[LimitType.TOKENS]:
            # If no specific config is set, allow the tokens (default behavior)
            return True, 0.0
        
        return self.limiters[LimitType.TOKENS][key].allow(count)
    
    def get_default_limiter(self, limit_type: LimitType, config: RateLimitConfig) -> GCRA:
        """
        Get a default limiter instance for when no specific user/model config exists.
        
        Args:
            limit_type: The type of limit (requests or tokens)
            config: The default configuration
            
        Returns:
            A GCRA instance with the provided configuration
        """
        return GCRA(config)