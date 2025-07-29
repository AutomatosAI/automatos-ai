
"""
Performance and load testing
"""
import pytest
import httpx
import asyncio
import time
from typing import List

@pytest.mark.asyncio
async def test_concurrent_health_checks(client: httpx.AsyncClient):
    """Test concurrent health check requests"""
    async def make_request():
        response = await client.get("/health")
        return response.status_code, response.elapsed.total_seconds()
    
    # Make 10 concurrent requests
    tasks = [make_request() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    status_codes = [result[0] for result in results]
    response_times = [result[1] for result in results]
    
    assert all(code == 200 for code in status_codes)
    assert max(response_times) < 5.0  # All responses under 5 seconds
    
    print(f"Concurrent requests - Max time: {max(response_times):.3f}s, Avg time: {sum(response_times)/len(response_times):.3f}s")

@pytest.mark.asyncio
async def test_api_response_times(client: httpx.AsyncClient):
    """Test response times for various endpoints"""
    endpoints = [
        "/health",
        "/",
        "/api/agents/",
        "/api/skills/",
        "/api/workflows/",
        "/listServers"
    ]
    
    response_times = {}
    
    for endpoint in endpoints:
        start_time = time.time()
        try:
            response = await client.get(endpoint)
            end_time = time.time()
            response_times[endpoint] = {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code < 400
            }
        except Exception as e:
            response_times[endpoint] = {
                "status_code": 0,
                "response_time": 0,
                "success": False,
                "error": str(e)
            }
    
    # Print performance summary
    print("\nAPI Performance Summary:")
    for endpoint, metrics in response_times.items():
        status = "✅" if metrics["success"] else "❌"
        print(f"{status} {endpoint}: {metrics['response_time']:.3f}s (HTTP {metrics['status_code']})")
    
    # At least health check should be fast
    assert response_times["/health"]["response_time"] < 1.0
    assert response_times["/health"]["success"]

@pytest.mark.asyncio
async def test_large_payload_handling(client: httpx.AsyncClient):
    """Test handling of large payloads"""
    # Create a large agent configuration
    large_config = {
        "name": "Large Config Agent",
        "description": "Agent with large configuration for testing",
        "agent_type": "custom",
        "configuration": {
            "large_data": "x" * 10000,  # 10KB of data
            "nested_config": {
                "level1": {"level2": {"level3": "deep_value"} for _ in range(100)}
            }
        }
    }
    
    start_time = time.time()
    response = await client.post("/api/agents/", json=large_config)
    end_time = time.time()
    
    print(f"Large payload test: {response.status_code} in {end_time - start_time:.3f}s")
    
    # Should handle large payloads (may fail due to validation, but shouldn't timeout)
    assert end_time - start_time < 10.0  # Should complete within 10 seconds
