import httpx
import json
import asyncio
import logging
import re
from typing import List, Dict, Any, AsyncGenerator, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from bot.rate_limiter import RateLimiter, RateLimitConfig, LimitType


class OpenRouterAPI:
    """
    Class to handle interactions with the OpenRouter API.
    Provides methods for sending chat completion requests with support for streaming.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1",
                 requests_per_minute: int = 60, tokens_per_minute: int = 100000,
                 tokens_per_day: int = 10000, concurrent_requests: int = 5):
        """
        Initialize the OpenRouter API client with the provided API key.
        
        Args:
            api_key: The API key for authenticating with OpenRouter
            base_url: The base URL for the OpenRouter API (default: https://openrouter.ai/api/v1)
            requests_per_minute: Maximum requests per minute allowed by the API
            tokens_per_minute: Maximum tokens per minute allowed by the API
            tokens_per_day: Maximum tokens per day allowed by the API
            concurrent_requests: Maximum concurrent requests allowed
        """
        self.api_key = api_key
        self.base_url = base_url
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_day = tokens_per_day
        self.concurrent_requests = concurrent_requests
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        # Set default configurations for rate limiting
        self.set_default_rate_limits()
        
        # For tracking concurrent requests
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0  # Set a reasonable timeout for API requests
        )
    
    def set_gcra_config_from_dict(self, user_id: str, model: str, limit_type: str, config_dict: dict):
        """
        Set GCRA configuration from a dictionary (used for loading from config file).
        
        Args:
            user_id: The user ID
            model: The model name
            limit_type: Either "requests" or "tokens"
            config_dict: Dictionary with "limit", "window", and "burst" keys
        """
        from bot.rate_limiter import LimitType, RateLimitConfig
        
        if not config_dict:
            return
            
        limit = config_dict.get("limit")
        window = config_dict.get("window", 60)  # Default to 60 seconds
        burst = config_dict.get("burst", 0)     # Default to 0 burst allowance
        
        if limit is not None:
            rate_limit_config = RateLimitConfig(limit=limit, window=window, burst=burst)
            limit_type_enum = LimitType.REQUESTS if limit_type == "requests" else LimitType.TOKENS
            self.rate_limiter.set_config(user_id, model, limit_type_enum, rate_limit_config)
    
    def set_default_rate_limits(self):
        """
        Set default rate limit configurations for the API.
        """
        # Default request rate limit: requests_per_minute
        request_config = RateLimitConfig(
            limit=self.requests_per_minute,
            window=60,  # 60 seconds
            burst=5     # Allow small burst
        )
        # Default token rate limit: tokens_per_minute
        token_config = RateLimitConfig(
            limit=self.tokens_per_minute,
            window=60,  # 60 seconds
            burst=self.tokens_per_minute // 10  # Allow 10% of limit as burst
        )
        
        # Set default configurations for user ID "default" and model "default"
        self.rate_limiter.set_config("default", "default", LimitType.REQUESTS, request_config)
        self.rate_limiter.set_config("default", "default", LimitType.TOKENS, token_config)
    
    def set_gcra_configs(self):
        """
        Set GCRA configurations from the provided config dictionaries.
        """
        # Set request rate limits for different models
        for model, config in self.gcra_requests_limit.items():
            self.set_gcra_config_from_dict("default", model, "requests", config)
        
        # Set token rate limits for different models
        for model, config in self.gcra_tokens_limit.items():
            self.set_gcra_config_from_dict("default", model, "tokens", config)
    
    def set_user_gcra_configs(self, user_id: str, gcra_settings: Dict[str, Any]):
        """
        Set GCRA configurations for a specific user from their settings dict.
        
        Args:
            user_id: The user ID
            gcra_settings: The user's GCRA settings dictionary
        """
        for model, settings in gcra_settings.items():
            if "requests" in settings:
                self.set_gcra_config_from_dict(user_id, model, "requests", settings["requests"])
            if "tokens" in settings:
                self.set_gcra_config_from_dict(user_id, model, "tokens", settings["tokens"])
    
    async def _check_rate_limits(self, user_id: str, model: str, estimated_tokens: int = 0) -> bool:
        """
        Check if the current request would exceed API rate limits using GCRA.
        
        Args:
            user_id: The ID of the user making the request
            model: The model being used for the request
            estimated_tokens: Estimated number of tokens for the upcoming request
            
        Returns:
            True if request is allowed, False if it would exceed limits
        """
        now = datetime.now()
        
        # Reset daily counter if we've moved to a new day
        if now.date() != self.today:
            self.tokens_today = 0
            self.today = now.date()
        
        # Check request rate limit using GCRA
        request_allowed, request_delay = self.rate_limiter.allow_request(user_id, model, 1)
        if not request_allowed:
            logging.warning(f"Request rate limit exceeded for user {user_id} with model {model}. "
                           f"Delay needed: {request_delay:.2f}s")
            return False
        
        # Check token rate limit using GCRA
        token_allowed, token_delay = self.rate_limiter.allow_tokens(user_id, model, estimated_tokens)
        if not token_allowed:
            logging.warning(f"Token rate limit exceeded for user {user_id} with model {model}. "
                           f"Delay needed: {token_delay:.2f}s")
            return False
        
        # Check daily token limit (this is still a simple counter-based limit)
        within_tokens_per_day_limit = self.tokens_today + estimated_tokens <= self.tokens_per_day
        if not within_tokens_per_day_limit:
            logging.warning(f"Daily token limit exceeded for user {user_id}. "
                           f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
            return False
        
        return True
    
    async def _update_usage(self, user_id: str, model: str, tokens_used: int):
        """
        Update the usage counters after a successful API request using GCRA.
        
        Args:
            user_id: The ID of the user making the request
            model: The model being used for the request
            tokens_used: Number of tokens used in the request
        """
        now = datetime.now()
        
        # Reset daily counter if we've moved to a new day
        if now.date() != self.today:
            self.tokens_today = 0
            self.today = now.date()
        
        # Update daily token counter (still needed for daily limit)
        self.tokens_today += tokens_used
    
    async def send_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = True,
        user_id: str = "default",  # Added user_id parameter for rate limiting
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a chat completion request to the OpenRouter API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model to use for completion (default: openai/gpt-3.5-turbo)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (optional)
            top_p: Nucleus sampling parameter (default: 1.0)
            frequency_penalty: Frequency penalty (default: 0.0)
            presence_penalty: Presence penalty (default: 0.0)
            stream: Whether to stream the response (default: True)
            user_id: The ID of the user making the request (default: "default")
            **kwargs: Additional parameters to pass to the API
             
        Yields:
            Dictionary containing the response chunk from the API
        """
        # Calculate estimated tokens for the request
        estimated_tokens = sum(len(msg.get("content", "")) for msg in messages) // 4  # Rough estimation
        if max_tokens:
            estimated_tokens += max_tokens
        else:
            estimated_tokens += self.tokens_per_minute // 100  # Default estimate if max_tokens not specified
        
        # Check rate limits before making the request
        if not await self._check_rate_limits(user_id, model, estimated_tokens):
            logging.warning(f"API rate limits would be exceeded for user {user_id} with model {model}. Request blocked.")
            raise Exception("API rate limits exceeded. Please try again later.")
        
        # Acquire semaphore for concurrent request limit
        async with self.semaphore:
            url = f"{self.base_url}/chat/completions"
            
            # Prepare the payload for the API request
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stream": stream
            }
            
            # Remove None values to avoid sending null parameters
            payload = {k: v for k, v in payload.items() if v is not None}
            
            # Add any additional parameters passed in kwargs
            payload.update(kwargs)
            
            # Maximum number of retries with different strategies for different error types
            max_retries = 3
            retry_delay = 2  # Initial delay in seconds
            max_retry_delay = 120  # Max delay in seconds to prevent excessively long waits
            
            for attempt in range(max_retries):
                try:
                    # Send the request to the API
                    async with self.client.stream("POST", url, json=payload) as response:
                        # Check if the request was successful
                        if response.status_code == 200:
                            # Process the streaming response
                            tokens_used = 0
                            async for line in response.aiter_lines():
                                if line.strip() and line.startswith("data: "):
                                    # Parse the data chunk
                                    data = line[6:]  # Remove "data: " prefix
                                    
                                    # Handle the special case of [DONE]
                                    if data.strip() == "[DONE]":
                                        break
                                    
                                    try:
                                        # Parse the JSON response
                                        json_data = json.loads(data)
                                        
                                        # Count tokens in the response if available
                                        if "usage" in json_data:
                                            usage = json_data["usage"]
                                            tokens_used = usage.get("total_tokens", 0)
                                        
                                        yield json_data
                                    except json.JSONDecodeError:
                                        # Skip invalid JSON lines
                                        continue
                            
                            # Update usage after successful request
                            await self._update_usage(user_id, model, tokens_used if tokens_used > 0 else estimated_tokens)
                            break # Exit the retry loop if successful
                        elif response.status_code == 429:
                            # Handle rate limit errors with exponential backoff
                            if attempt == max_retries - 1:
                                logging.warning(f"Max retries exceeded for 429 error. Current usage: "
                                               f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                               f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                               f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                                raise Exception("Rate limit exceeded. Please try again later.")
                            
                            # Calculate delay with more conservative exponential backoff (2s, 6s, 18s, ...) with jitter
                            delay = min(retry_delay * (3 ** attempt), max_retry_delay)
                            # Add jitter to prevent thundering herd problem
                            import random
                            delay = delay * (0.7 + random.random() * 0.3)
                            
                            logging.warning(f"Rate limit exceeded (429) on attempt {attempt + 1}, retrying after {delay:.2f}s. Current usage: "
                                           f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                           f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                           f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                            await asyncio.sleep(delay)
                        elif response.status_code == 500:
                            # Handle internal server errors with longer delays
                            if attempt == max_retries - 1:
                                error_text = await response.aread()
                                logging.error(f"API request failed with status 50 after {max_retries} attempts: {error_text}. Current usage: "
                                             f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                             f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                             f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                                raise Exception(f"API internal server error (500): {error_text.decode('utf-8', errors='ignore')}")
                            
                            # Use longer delay for 500 errors
                            delay = min(retry_delay * (3 ** attempt), max_retry_delay)
                            import random
                            delay = delay * (0.7 + random.random() * 0.3)
                            
                            error_text = await response.aread()
                            logging.warning(f"API internal server error (500) on attempt {attempt + 1}, retrying after {delay:.2f}s. Error: {error_text.decode('utf-8', errors='ignore')}. Current usage: "
                                           f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                           f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                           f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                            await asyncio.sleep(delay)
                        elif response.status_code in [502, 503, 504]:
                            # Handle gateway errors with exponential backoff
                            if attempt == max_retries - 1:
                                error_text = await response.aread()
                                logging.error(f"API gateway error ({response.status_code}) after {max_retries} attempts: {error_text}. Current usage: "
                                             f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                             f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                             f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                                raise Exception(f"API gateway error ({response.status_code}): {error_text.decode('utf-8', errors='ignore')}")
                            
                            # Use moderate delay for gateway errors
                            delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                            import random
                            delay = delay * (0.7 + random.random() * 0.3)
                            
                            error_text = await response.aread()
                            logging.warning(f"API gateway error ({response.status_code}) on attempt {attempt + 1}, retrying after {delay:.2f}s. Error: {error_text.decode('utf-8', errors='ignore')}. Current usage: "
                                           f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                           f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                           f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                            await asyncio.sleep(delay)
                        elif response.status_code >= 400:
                            # Handle other client/server errors (4xx, 5xx)
                            error_text = await response.aread()
                            error_message = error_text.decode('utf-8', errors='ignore')
                            
                            # Don't retry for certain client errors (4xx)
                            if 400 <= response.status_code < 500 and response.status_code not in [429]:
                                logging.error(f"API request failed with status {response.status_code}: {error_message}. This is a client error, not retrying. Current usage: "
                                             f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                             f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                             f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                                raise Exception(f"API request failed with status {response.status_code}: {error_message}")
                            
                            # Retry for server errors (5xx except 500 which is handled above)
                            if attempt == max_retries - 1:
                                logging.error(f"API request failed with status {response.status_code} after {max_retries} attempts: {error_message}. Current usage: "
                                             f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                             f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                             f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                                raise Exception(f"API request failed with status {response.status_code}: {error_message}")
                            
                            # Use exponential backoff for other errors
                            delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                            import random
                            delay = delay * (0.7 + random.random() * 0.3)
                            
                            logging.warning(f"API request failed with status {response.status_code} on attempt {attempt + 1}, retrying after {delay:.2f}s. Error: {error_message}. Current usage: "
                                           f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                           f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                           f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                            await asyncio.sleep(delay)
                        else:
                            # Handle unexpected status codes
                            error_text = await response.aread()
                            error_message = error_text.decode('utf-8', errors='ignore')
                            
                            if attempt == max_retries - 1:
                                logging.error(f"Unexpected API response status {response.status_code}: {error_message}. Current usage: "
                                             f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                             f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                             f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                                raise Exception(f"Unexpected API response status {response.status_code}: {error_message}")
                            
                            # Retry for unexpected status codes with moderate delay
                            delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                            import random
                            delay = delay * (0.7 + random.random() * 0.3)
                            
                            logging.warning(f"Unexpected API response status {response.status_code} on attempt {attempt + 1}, retrying after {delay:.2f}s. Error: {error_message}. Current usage: "
                                           f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                           f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                           f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                            await asyncio.sleep(delay)
                
                except httpx.TimeoutException as e:
                    # Handle timeout errors specifically
                    if attempt == max_retries - 1:
                        logging.error(f"Request timeout after {max_retries} attempts. Current usage: "
                                     f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                     f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                     f"Tokens today: {self.tokens_today}/{self.tokens_per_day}. Error: {str(e)}")
                        raise Exception(f"Request timeout: {str(e)}")
                    
                    # Use exponential backoff for timeout errors
                    delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                    import random
                    delay = delay * (0.7 + random.random() * 0.3)
                    
                    logging.warning(f"Request timeout on attempt {attempt + 1}, retrying after {delay:.2f}s: {str(e)}. Current usage: "
                                   f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                   f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                   f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                    await asyncio.sleep(delay)
                
                except httpx.RequestError as e:
                    # Handle connection and other request errors
                    if attempt == max_retries - 1:
                        logging.error(f"Request failed after {max_retries} attempts due to connection error. Current usage: "
                                     f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                     f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                     f"Tokens today: {self.tokens_today}/{self.tokens_per_day}. Error: {str(e)}")
                        raise Exception(f"Request failed due to connection error: {str(e)}")
                    
                    # Use exponential backoff for connection errors
                    delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                    import random
                    delay = delay * (0.7 + random.random() * 0.3)
                    
                    logging.warning(f"Request failed on attempt {attempt + 1} due to connection error, retrying after {delay:.2f}s: {str(e)}. Current usage: "
                                   f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                   f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                   f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                    await asyncio.sleep(delay)
                
                except Exception as e:
                    # Handle any other unexpected exceptions
                    if attempt == max_retries - 1:
                        logging.error(f"Unexpected error after {max_retries} attempts. Current usage: "
                                     f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                     f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                     f"Tokens today: {self.tokens_today}/{self.tokens_per_day}. Error: {str(e)}")
                        raise e
                    
                    # Use exponential backoff for other exceptions
                    delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                    import random
                    delay = delay * (0.7 + random.random() * 0.3)
                    
                    logging.warning(f"Unexpected error on attempt {attempt + 1}, retrying after {delay:.2f}s: {str(e)}. Current usage: "
                                   f"Requests this minute: {self.requests_this_minute}/{self.requests_per_minute}, "
                                   f"Tokens this minute: {self.tokens_this_minute}/{self.tokens_per_minute}, "
                                   f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
                    await asyncio.sleep(delay)
    
    async def close(self):
        """
        Close the HTTP client to free up resources.
        """
        await self.client.aclose()


# Example usage function
async def example_usage():
    """
    Example of how to use the OpenRouterAPI class.
    This is just for demonstration purposes.
    """
    # Initialize the API client with your API key
    api = OpenRouterAPI(
        api_key="your-api-key-here",
        requests_per_minute=60,
        tokens_per_minute=100000,
        tokens_per_day=1000,
        concurrent_requests=5
    )
    
    try:
        # Define the messages for the conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        # Send the chat completion request and process the response
        async for chunk in api.send_chat_completion(
            messages=messages,
            model="openai/gpt-3.5-turbo",
            temperature=0.7
        ):
            # Print the content of each chunk
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    print(delta["content"], end="", flush=True)
        
        print()  # New line after the response is complete
    
    except Exception as e:
        logging.error(f"Error in example usage: {e}")
    
    finally:
        # Close the API client to free up resources
        await api.close()


class GoogleAIAPI:
    """
    Class to handle interactions with the Google AI API (Gemini).
    Provides methods for sending chat completion requests with support for streaming.
    """
    
    def __init__(self, api_key: str, requests_per_minute: int = 60, tokens_per_minute: int = 100000,
                 tokens_per_day: int = 10000, concurrent_requests: int = 5):
        """
        Initialize the Google AI API client with the provided API key.
        
        Args:
            api_key: The API key for authenticating with Google AI
            requests_per_minute: Maximum requests per minute allowed by the API
            tokens_per_minute: Maximum tokens per minute allowed by the API
            tokens_per_day: Maximum tokens per day allowed by the API
            concurrent_requests: Maximum concurrent requests allowed
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_day = tokens_per_day
        self.concurrent_requests = concurrent_requests
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        # Set default configurations for rate limiting
        self.set_default_rate_limits()
        
        # For tracking concurrent requests
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        
        # Initialize the model (we'll set the specific model in the method call)
        self.models = {}  # Cache for different models

    def set_gcra_config_from_dict(self, user_id: str, model: str, limit_type: str, config_dict: dict):
        """
        Set GCRA configuration from a dictionary (used for loading from config file).
        
        Args:
            user_id: The user ID
            model: The model name
            limit_type: Either "requests" or "tokens"
            config_dict: Dictionary with "limit", "window", and "burst" keys
        """
        from bot.rate_limiter import LimitType, RateLimitConfig
        
        if not config_dict:
            return
            
        limit = config_dict.get("limit")
        window = config_dict.get("window", 60)  # Default to 60 seconds
        burst = config_dict.get("burst", 0)     # Default to 0 burst allowance
        
        if limit is not None:
            rate_limit_config = RateLimitConfig(limit=limit, window=window, burst=burst)
            limit_type_enum = LimitType.REQUESTS if limit_type == "requests" else LimitType.TOKENS
            self.rate_limiter.set_config(user_id, model, limit_type_enum, rate_limit_config)

    def set_default_rate_limits(self):
        """
        Set default rate limit configurations for the API.
        """
        # Default request rate limit: requests_per_minute
        request_config = RateLimitConfig(
            limit=self.requests_per_minute,
            window=60, # 60 seconds
            burst=5     # Allow small burst
        )
        # Default token rate limit: tokens_per_minute
        token_config = RateLimitConfig(
            limit=self.tokens_per_minute,
            window=60,  # 60 seconds
            burst=self.tokens_per_minute // 10  # Allow 10% of limit as burst
        )
        
        # Set default configurations for user ID "default" and model "default"
        self.rate_limiter.set_config("default", "default", LimitType.REQUESTS, request_config)
        self.rate_limiter.set_config("default", "default", LimitType.TOKENS, token_config)

    def set_user_gcra_configs(self, user_id: str, gcra_settings: Dict[str, Any]):
        """
        Set GCRA configurations for a specific user from their settings dict.
        
        Args:
            user_id: The user ID
            gcra_settings: The user's GCRA settings dictionary
        """
        for model, settings in gcra_settings.items():
            if "requests" in settings:
                self.set_gcra_config_from_dict(user_id, model, "requests", settings["requests"])
            if "tokens" in settings:
                self.set_gcra_config_from_dict(user_id, model, "tokens", settings["tokens"])

    async def _check_rate_limits(self, user_id: str, model: str, estimated_tokens: int = 0) -> bool:
        """
        Check if the current request would exceed API rate limits using GCRA.
        
        Args:
            user_id: The ID of the user making the request
            model: The model being used for the request
            estimated_tokens: Estimated number of tokens for the upcoming request
            
        Returns:
            True if request is allowed, False if it would exceed limits
        """
        now = datetime.now()
        
        # Reset daily counter if we've moved to a new day
        if now.date() != self.today:
            self.tokens_today = 0
            self.today = now.date()
        
        # Check request rate limit using GCRA
        request_allowed, request_delay = self.rate_limiter.allow_request(user_id, model, 1)
        if not request_allowed:
            logging.warning(f"Request rate limit exceeded for user {user_id} with model {model}. "
                           f"Delay needed: {request_delay:.2f}s")
            return False
        
        # Check token rate limit using GCRA
        token_allowed, token_delay = self.rate_limiter.allow_tokens(user_id, model, estimated_tokens)
        if not token_allowed:
            logging.warning(f"Token rate limit exceeded for user {user_id} with model {model}. "
                           f"Delay needed: {token_delay:.2f}s")
            return False
        
        # Check daily token limit (this is still a simple counter-based limit)
        within_tokens_per_day_limit = self.tokens_today + estimated_tokens <= self.tokens_per_day
        if not within_tokens_per_day_limit:
            logging.warning(f"Daily token limit exceeded for user {user_id}. "
                           f"Tokens today: {self.tokens_today}/{self.tokens_per_day}")
            return False
        
        return True

    async def _update_usage(self, user_id: str, model: str, tokens_used: int):
        """
        Update the usage counters after a successful API request using GCRA.
        
        Args:
            user_id: The ID of the user making the request
            model: The model being used for the request
            tokens_used: Number of tokens used in the request
        """
        now = datetime.now()
        
        # Reset daily counter if we've moved to a new day
        if now.date() != self.today:
            self.tokens_today = 0
            self.today = now.date()
        
        # Update daily token counter (still needed for daily limit)
        self.tokens_today += tokens_used

    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert messages from OpenAI format to Google AI format.
        In Google AI format, 'assistant' role becomes 'model'.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            
        Returns:
            List of message dictionaries in Google AI format
        """
        converted_messages = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            # Convert role from OpenAI format to Google AI format
            if role == "assistant":
                converted_messages.append({"role": "model", "parts": [{"text": content}] if content else [{"text": ""}]})
            else:
                converted_messages.append({"role": role, "parts": [{"text": content}] if content else [{"text": ""}]})
        return converted_messages

    async def send_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "google/gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = True,
        user_id: str = "default",  # Added user_id parameter for rate limiting
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a chat completion request to the Google AI API using the Gemini model.
        Supports streaming responses via streamGenerateContent endpoint.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model to use for completion (default: google/gemini-2.0-flash-exp)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (optional)
            top_p: Nucleus sampling parameter (default: 1.0)
            frequency_penalty: Frequency penalty (default: 0.0)
            presence_penalty: Presence penalty (default: 0.0)
            stream: Whether to stream the response (default: True)
            user_id: The ID of the user making the request (default: "default")
            **kwargs: Additional parameters to pass to the API
             
        Yields:
            Dictionary containing the response chunk from the API
        """
        # Calculate estimated tokens for the request
        estimated_tokens = sum(len(msg.get("content", "")) for msg in messages) // 4  # Rough estimation
        if max_tokens:
            estimated_tokens += max_tokens
        else:
            estimated_tokens += self.tokens_per_minute // 100  # Default estimate if max_tokens not specified
        
        # Check rate limits before making the request
        if not await self._check_rate_limits(user_id, model, estimated_tokens):
            logging.warning(f"API rate limits would be exceeded for user {user_id} with model {model}. Request blocked.")
            raise Exception("API rate limits exceeded. Please try again later.")
        
        # Acquire semaphore for concurrent request limit
        async with self.semaphore:
            # Get or create the model instance
            if model not in self.models:
                # Remove provider prefix if present to get the actual model name for Google AI
                actual_model_name = model.replace("google/", "")
                self.models[model] = genai.GenerativeModel(actual_model_name)
            model_instance = self.models[model]
            
            # Convert messages from OpenAI format to Google AI format
            google_ai_messages = self._convert_messages_format(messages)
            
            # Prepare the generation configuration
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
                "candidate_count": 1,
            }
            
            # Remove None values to avoid sending null parameters
            generation_config = {k: v for k, v in generation_config.items() if v is not None}
            
            try:
                # Handle streaming vs non-streaming requests
                if stream:
                    # Use the streaming method
                    response = await model_instance.generate_content_async(
                        google_ai_messages,
                        generation_config=generation_config,
                        stream=True
                    )
                    
                    # Process the streaming response
                    tokens_used = 0
                    async for chunk in response:
                        # Convert the chunk to the expected format
                        # Google AI SDK returns parts in a different format than OpenAI
                        if hasattr(chunk, 'text') and chunk.text:
                            # Create a response in OpenAI-compatible format
                            response_chunk = {
                                "candidates": [{
                                    "content": {
                                        "parts": [{"text": chunk.text}],
                                        "role": "model"
                                    },
                                    "finish_reason": None,
                                    "index": 0,
                                    "safety_ratings": getattr(chunk, 'safety_ratings', [])
                                }],
                                "prompt_feedback": getattr(chunk, 'prompt_feedback', {})
                            }
                            yield response_chunk
                            
                            # Count tokens in the response if available
                            tokens_used += len(chunk.text) // 4  # Rough estimation
                        else:
                            # Handle other response types if needed
                            response_chunk = {
                                "candidates": [{
                                    "content": {
                                        "parts": [],
                                        "role": "model"
                                    },
                                    "finish_reason": "stop",
                                    "index": 0,
                                    "safety_ratings": getattr(chunk, 'safety_ratings', [])
                                }],
                                "prompt_feedback": getattr(chunk, 'prompt_feedback', {})
                            }
                            yield response_chunk
                else:
                    # Non-streaming request
                    response = await model_instance.generate_content_async(
                        google_ai_messages,
                        generation_config=generation_config,
                        stream=False
                    )
                    
                    # Convert the response to the expected format
                    if response.text:
                        response_chunk = {
                            "candidates": [{
                                "content": {
                                    "parts": [{"text": response.text}],
                                    "role": "model"
                                },
                                "finish_reason": "stop",
                                "index": 0,
                                "safety_ratings": getattr(response, 'safety_ratings', [])
                            }],
                            "prompt_feedback": getattr(response, 'prompt_feedback', {})
                        }
                        yield response_chunk
                        
                        # Count tokens in the response if available
                        tokens_used = len(response.text) // 4  # Rough estimation

                # Update usage after successful request
                await self._update_usage(user_id, model, tokens_used if tokens_used > 0 else estimated_tokens)

            except Exception as e:
                logging.error(f"Error calling Google AI API: {str(e)}")
                raise e

    async def close(self):
        """
        Close the API client to free up resources.
        """
        # Nothing specific needed for Google AI API client
        pass
# Example usage function for GoogleAIAPI
async def example_usage_google():
    """
    Example of how to use the GoogleAIAPI class.
    This is just for demonstration purposes.
    """
    # Initialize the API client with your API key
    api = GoogleAIAPI(
        api_key="your-google-api-key-here",
        requests_per_minute=60,
        tokens_per_minute=100000,
        tokens_per_day=100,
        concurrent_requests=5
    )
    
    try:
        # Define the messages for the conversation
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        # Send the chat completion request and process the response
        async for chunk in api.send_chat_completion(
            messages=messages,
            model="google/gemini-2.0-flash-exp",
            temperature=0.7
        ):
            # Print the content of each chunk
            if "candidates" in chunk and len(chunk["candidates"]) > 0:
                content = chunk["candidates"][0].get("content", {})
                if "parts" in content and len(content["parts"]) > 0:
                    text = content["parts"][0].get("text", "")
                    if text:
                        print(text, end="", flush=True)
        
        print()  # New line after the response is complete
    
    except Exception as e:
        logging.error(f"Error in Google API example usage: {e}")
    
    finally:
        # Close the API client to free up resources
        await api.close()


# This allows running the example when the module is executed directly
if __name__ == "__main__":
    # Run the example (this will fail without a real API key)
    asyncio.run(example_usage_google())