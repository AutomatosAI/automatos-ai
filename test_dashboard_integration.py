#!/usr/bin/env python3
"""
Test script to verify PRD06 Dashboard Integration
Tests the analytics API and WebSocket functionality
"""

import requests
import json
import time
import websocket
import threading

def test_analytics_api():
    """Test the analytics API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing PRD06 Dashboard Analytics API...")
    
    # Test dashboard overview
    try:
        response = requests.get(f"{base_url}/api/analytics/dashboard/overview", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Dashboard Overview API working")
            print(f"   - System Health: CPU {data['systemHealth']['cpuUsage']}%, Memory {data['systemHealth']['memoryUsage']}%")
            print(f"   - Redis Status: {data['systemHealth']['redisStatus']}")
            print(f"   - Active Agents: {data['agentMetrics']['activeAgents']}/{data['agentMetrics']['totalAgents']}")
            print(f"   - Total Workflows: {data['workflowMetrics']['totalWorkflows']}")
        else:
            print(f"❌ Dashboard Overview API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard Overview API error: {e}")
    
    # Test individual endpoints
    endpoints = [
        "/api/analytics/system/health",
        "/api/analytics/agents", 
        "/api/analytics/workflows",
        "/api/analytics/context",
        "/api/analytics/learning"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} working")
            else:
                print(f"❌ {endpoint} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} error: {e}")

def test_websocket():
    """Test WebSocket connection"""
    print("\n🔌 Testing WebSocket connection...")
    
    def on_message(ws, message):
        print(f"📨 WebSocket message received: {message[:100]}...")
    
    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("🔌 WebSocket connection closed")
    
    def on_open(ws):
        print("✅ WebSocket connection opened")
        # Send a ping
        ws.send("ping")
    
    try:
        ws = websocket.WebSocketApp("ws://localhost:8000/ws/dashboard",
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
        
        # Run WebSocket in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Wait for connection and messages
        time.sleep(3)
        
        # Close the connection
        ws.close()
        print("✅ WebSocket test completed")
        
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

def test_frontend_integration():
    """Test if frontend can access the analytics API"""
    print("\n🌐 Testing Frontend Integration...")
    
    try:
        # Test if frontend is running
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is running")
            
            # Check if the page contains dashboard elements
            content = response.text
            if "dashboard" in content.lower() or "analytics" in content.lower():
                print("✅ Frontend contains dashboard elements")
            else:
                print("⚠️  Frontend may not have dashboard integration")
        else:
            print(f"❌ Frontend not accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")

def main():
    print("🚀 PRD06 Dashboard Integration Test")
    print("=" * 50)
    
    test_analytics_api()
    test_websocket()
    test_frontend_integration()
    
    print("\n" + "=" * 50)
    print("✅ Dashboard integration test completed!")
    print("\n📊 Dashboard should now show:")
    print("   - Real-time system metrics (CPU, Memory, Disk)")
    print("   - Live agent status and performance")
    print("   - Workflow execution analytics")
    print("   - Context optimization metrics")
    print("   - Learning and memory insights")
    print("   - WebSocket real-time updates")

if __name__ == "__main__":
    main()
