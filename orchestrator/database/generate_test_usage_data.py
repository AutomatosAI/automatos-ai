#!/usr/bin/env python3
"""
Generate test usage data for Context Engineering page via API
This creates realistic tracking data to verify the analytics system works
"""

import requests
import json
import time
import random
from datetime import datetime, timedelta

# API Configuration
#API_BASE_URL = "http://ui.automatos.app"  # Production URL (if using nginx proxy)
API_BASE_URL = "http://206.81.0.227:8000"  # Direct IP
DOCUMENTS_API = f"{API_BASE_URL}/api/documents/v2"

# Test data
SAMPLE_QUERIES = [
    "How do I configure authentication?",
    "What is the deployment process?",
    "Database optimization strategies",
    "Error handling best practices",
    "API rate limiting implementation",
    "Caching strategies for performance",
    "Security best practices guide",
    "Microservices architecture patterns",
    "Docker container optimization",
    "CI/CD pipeline setup"
]

def generate_search_events(count=20):
    """Generate document search events with realistic patterns"""
    print(f"\n🔍 Generating {count} search events...")
    
    for i in range(count):
        query = random.choice(SAMPLE_QUERIES)
        results_count = random.randint(0, 15)  # Some searches return no results
        execution_time = random.randint(50, 500)  # 50-500ms
        
        # Simulate search via API (which auto-tracks in document_usage)
        try:
            response = requests.post(
                f"{DOCUMENTS_API}/search",
                params={
                    "query": query,
                    "limit": 10,
                    "min_similarity": 0.7
                },
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"  ✅ Search tracked: '{query[:50]}' ({results_count} results, {execution_time}ms)")
            else:
                print(f"  ⚠️  Search failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Small delay to simulate realistic usage
        time.sleep(random.uniform(0.5, 2.0))

def generate_rag_events(count=10):
    """Generate RAG query events"""
    print(f"\n🧠 Generating {count} RAG query events...")
    
    for i in range(count):
        query = random.choice(SAMPLE_QUERIES)
        
        try:
            response = requests.post(
                f"{DOCUMENTS_API}/rag/retrieve",
                params={
                    "query": query,
                    "max_chunks": random.randint(3, 10),
                    "max_tokens": random.randint(1000, 4000)
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                chunks = len(data.get('chunks', []))
                exec_time = data.get('execution_time_ms', 0)
                print(f"  ✅ RAG query tracked: '{query[:50]}' ({chunks} chunks, {exec_time}ms)")
            else:
                print(f"  ⚠️  RAG failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        time.sleep(random.uniform(1.0, 3.0))

def generate_manual_tracking_events(count=15):
    """Generate manual tracking events using the analytics API"""
    print(f"\n📊 Generating {count} manual tracking events...")
    
    event_types = ['document_viewed', 'document_searched', 'chunk_retrieved', 'rag_query']
    
    for i in range(count):
        event_type = random.choice(event_types)
        
        try:
            response = requests.post(
                f"{DOCUMENTS_API}/analytics/track",
                params={
                    "event_type": event_type,
                    "query": random.choice(SAMPLE_QUERIES) if 'search' in event_type or 'rag' in event_type else None,
                    "metadata": json.dumps({
                        "test_data": True,
                        "generated_at": datetime.now().isoformat()
                    })
                },
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"  ✅ Event tracked: {event_type}")
            else:
                print(f"  ⚠️  Track failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        time.sleep(random.uniform(0.3, 1.0))

def verify_data():
    """Verify the generated data via API"""
    print("\n🔍 Verifying generated data...")
    
    try:
        # Check context stats
        response = requests.get(f"{API_BASE_URL}/api/context/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"  📊 Context Stats:")
            print(f"     - Queries: {stats.get('contextQueries', 0)}")
            print(f"     - Success Rate: {stats.get('retrievalSuccess', 0)}%")
            print(f"     - Avg Response Time: {stats.get('avgResponseTime', 'N/A')}")
            print(f"     - Vector Embeddings: {stats.get('vectorEmbeddings', 0)}")
        
        # Check recent queries
        response = requests.get(f"{API_BASE_URL}/api/context/queries/recent?limit=5", timeout=5)
        if response.status_code == 200:
            queries = response.json()
            print(f"\n  📋 Recent Queries: {len(queries)} found")
            for q in queries[:3]:
                print(f"     - {q.get('query', 'N/A')[:50]}")
        
    except Exception as e:
        print(f"  ❌ Verification error: {e}")

def main():
    print("=" * 70)
    print("🚀 Context Engineering Test Data Generator")
    print("=" * 70)
    print(f"\nAPI Base URL: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Test API connectivity
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("\n❌ API is not responding. Check if the server is running.")
            return
        print("\n✅ API is responding")
        
    except Exception as e:
        print(f"\n❌ Cannot connect to API: {e}")
        print("   Make sure the orchestrator is running on 206.81.0.227:8000")
        return
    
    # Generate different types of events
    generate_search_events(count=15)
    generate_rag_events(count=8)
    generate_manual_tracking_events(count=12)
    
    # Verify the generated data
    verify_data()
    
    print("\n" + "=" * 70)
    print("✅ Test data generation complete!")
    print("=" * 70)
    print("\n📈 Next steps:")
    print("   1. Open the Context Engineering page in the UI")
    print("   2. Verify charts show real data (not zeros)")
    print("   3. Check Recent Queries tab shows the test queries")
    print("   4. Check Performance tab shows metrics")
    print("")

if __name__ == "__main__":
    main()

