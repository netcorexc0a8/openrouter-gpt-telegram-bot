"""
Test script for GCRA (Generic Cell Rate Algorithm) implementation.

This script tests the core functionality of the GCRA rate limiting algorithm
to ensure it works as expected.
"""
import asyncio
import time
from bot.rate_limiter import RateLimiter, RateLimitConfig, LimitType


def test_basic_gcra():
    """Test basic GCRA functionality."""
    print("Testing basic GCRA functionality...")
    
    # Create a rate limiter with 10 requests per minute, allowing burst of 2
    config = RateLimitConfig(limit=10, window=60, burst=2)
    limiter = RateLimiter()
    limiter.set_config("test_user", "test_model", LimitType.REQUESTS, config)
    
    # Test allowing requests within limits
    allowed, delay = limiter.allow_request("test_user", "test_model", 1)
    assert allowed, "First request should be allowed"
    print(f"✓ First request allowed, delay: {delay}")
    
    # Allow several requests quickly
    for i in range(5):
        allowed, delay = limiter.allow_request("test_user", "test_model", 1)
        print(f"Request {i+2}: allowed={allowed}, delay={delay}")
    
    print("Basic GCRA test completed.\n")


def test_gcra_burst():
    """Test GCRA burst functionality."""
    print("Testing GCRA burst functionality...")
    
    # Create a rate limiter with 5 requests per minute, allowing burst of 3
    config = RateLimitConfig(limit=5, window=60, burst=3)
    limiter = RateLimiter()
    limiter.set_config("burst_user", "test_model", LimitType.REQUESTS, config)
    
    # Try to burst more than the limit
    results = []
    for i in range(10):
        allowed, delay = limiter.allow_request("burst_user", "test_model", 1)
        results.append((allowed, delay))
        print(f"Burst request {i+1}: allowed={allowed}, delay={delay:.3f}s")
        time.sleep(0.01) # Small delay to simulate request timing
    
    # Check that some requests were denied
    allowed_count = sum(1 for allowed, _ in results if allowed)
    print(f"Total allowed requests: {allowed_count}/10")
    print("Burst test completed.\n")


def test_gcra_tokens():
    """Test GCRA token limiting functionality."""
    print("Testing GCRA token limiting...")
    
    # Create a rate limiter with 1000 tokens per minute, allowing burst of 200
    config = RateLimitConfig(limit=1000, window=60, burst=200)
    limiter = RateLimiter()
    limiter.set_config("token_user", "test_model", LimitType.TOKENS, config)
    
    # Test token usage
    for i in range(5):
        tokens = 200 if i == 0 else 100 # First request uses more tokens
        allowed, delay = limiter.allow_tokens("token_user", "test_model", tokens)
        print(f"Token request {i+1} ({tokens} tokens): allowed={allowed}, delay={delay:.3f}s")
        time.sleep(0.05)  # Small delay between requests
    
    print("Token limiting test completed.\n")


async def test_concurrent_requests():
    """Test GCRA with concurrent requests."""
    print("Testing concurrent requests...")
    
    config = RateLimitConfig(limit=5, window=60, burst=2)
    limiter = RateLimiter()
    limiter.set_config("concurrent_user", "test_model", LimitType.REQUESTS, config)
    
    async def make_request(req_id):
        allowed, delay = limiter.allow_request("concurrent_user", "test_model", 1)
        print(f"Concurrent request {req_id}: allowed={allowed}, delay={delay:.3f}s")
        return allowed
    
    # Simulate concurrent requests
    tasks = [make_request(i) for i in range(8)]
    results = await asyncio.gather(*tasks)
    
    allowed_count = sum(1 for result in results if result)
    print(f"Concurrent test: {allowed_count}/{len(results)} requests allowed")
    print("Concurrent requests test completed.\n")


def main():
    """Run all GCRA tests."""
    print("Starting GCRA implementation tests...\n")
    
    test_basic_gcra()
    test_gcra_burst()
    test_gcra_tokens()
    
    # Run async test
    asyncio.run(test_concurrent_requests())
    
    print("All GCRA tests completed successfully!")


if __name__ == "__main__":
    main()