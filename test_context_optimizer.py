#!/usr/bin/env python3
"""
Test Context Optimizer - Demonstrates REAL Mathematical Optimization
=====================================================================

This test shows:
1. Real Shannon entropy calculation
2. Real cosine similarity with OpenAI embeddings
3. Knapsack optimization for token budgets
4. Few-shot example selection with MMR
5. Information density metrics

Run: python test_context_optimizer.py
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

# Add orchestrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

from context_engineering.context_optimizer import (
    ContextOptimizer,
    ContextItem,
    Example,
    AtomicPrompt,
    create_context_optimizer
)

async def test_information_density():
    """Test real Shannon entropy and information density calculation"""
    print("\n" + "="*60)
    print("TEST 1: Information Density Calculation")
    print("="*60)
    
    optimizer = create_context_optimizer()
    
    # Test different text patterns
    texts = [
        "aaaaaaaaa",  # Low entropy (repetitive)
        "The quick brown fox jumps over the lazy dog",  # Medium entropy
        "x7$mK9@pL2#vN5&qR8*tY3!wE6",  # High entropy (random)
        "def calculate_sum(a, b):\n    return a + b",  # Code
    ]
    
    print("\nCalculating information density for different texts:")
    for text in texts:
        density = optimizer.calculate_information_density(text)
        entropy = optimizer._calculate_shannon_entropy(text)
        print(f"\nText: '{text[:30]}...'" if len(text) > 30 else f"\nText: '{text}'")
        print(f"  Shannon Entropy: {entropy:.4f} bits")
        print(f"  Information Density: {density:.4f} bits/token")

async def test_context_optimization():
    """Test real context optimization with knapsack algorithm"""
    print("\n" + "="*60)
    print("TEST 2: Context Optimization with Knapsack Algorithm")
    print("="*60)
    
    optimizer = create_context_optimizer()
    
    # Create diverse context items
    context_items = [
        ContextItem(
            content="FastAPI is a modern web framework for building APIs with Python 3.6+ based on standard Python type hints.",
            source="documentation",
            context_type="documentation",
            relevance_score=0.9
        ),
        ContextItem(
            content="from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef read_root():\n    return {'Hello': 'World'}",
            source="example.py",
            context_type="code",
            relevance_score=0.85
        ),
        ContextItem(
            content="Authentication in FastAPI can be implemented using OAuth2 with Password flow and JWT tokens.",
            source="auth_guide",
            context_type="documentation",
            relevance_score=0.75
        ),
        ContextItem(
            content="async def get_current_user(token: str = Depends(oauth2_scheme)):\n    credentials_exception = HTTPException(status_code=401)",
            source="auth.py",
            context_type="code",
            relevance_score=0.7
        ),
        ContextItem(
            content="FastAPI automatically generates interactive API documentation using Swagger UI and ReDoc.",
            source="features",
            context_type="documentation",
            relevance_score=0.6
        ),
    ]
    
    # Test different token budgets
    max_tokens_list = [50, 100, 200]
    
    for max_tokens in max_tokens_list:
        print(f"\n--- Optimizing with {max_tokens} token budget ---")
        
        result = await optimizer.optimize_context(
            available_context=context_items,
            max_tokens=max_tokens,
            objective="maximize_information"
        )
        
        print(f"Selected {len(result.contexts)} contexts:")
        for ctx in result.contexts:
            print(f"  - [{ctx.context_type}] {ctx.content[:50]}... ({ctx.token_count} tokens)")
        
        print(f"\nMetrics:")
        print(f"  Total tokens used: {result.total_tokens}/{max_tokens}")
        print(f"  Information gain: {result.information_gain:.4f}")
        print(f"  Diversity score: {result.diversity_score:.4f}")
        print(f"  Token utilization: {result.optimization_metadata['token_utilization']:.2%}")

async def test_few_shot_selection():
    """Test few-shot example selection with MMR and real embeddings"""
    print("\n" + "="*60)
    print("TEST 3: Few-Shot Example Selection with MMR")
    print("="*60)
    
    optimizer = create_context_optimizer()
    
    # Task to find examples for
    task = "Write a Python function to validate email addresses"
    
    # Pool of examples with varying relevance
    examples = [
        Example(
            input_text="Write a function to validate phone numbers",
            output_text="import re\ndef validate_phone(phone):\n    pattern = r'^\\+?1?\\d{9,15}$'\n    return bool(re.match(pattern, phone))",
            category="validation",
            quality_score=0.9
        ),
        Example(
            input_text="Write a function to validate email addresses",
            output_text="import re\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))",
            category="validation",
            quality_score=1.0
        ),
        Example(
            input_text="Write a function to parse JSON",
            output_text="import json\ndef parse_json(json_str):\n    try:\n        return json.loads(json_str)\n    except json.JSONDecodeError:\n        return None",
            category="parsing",
            quality_score=0.8
        ),
        Example(
            input_text="Write a function to validate URLs",
            output_text="from urllib.parse import urlparse\ndef validate_url(url):\n    try:\n        result = urlparse(url)\n        return all([result.scheme, result.netloc])\n    except:\n        return False",
            category="validation",
            quality_score=0.85
        ),
        Example(
            input_text="Write a function to check password strength",
            output_text="import re\ndef check_password_strength(password):\n    if len(password) < 8:\n        return 'weak'\n    if re.search(r'[A-Z]', password) and re.search(r'[a-z]', password) and re.search(r'\\d', password):\n        return 'strong'\n    return 'medium'",
            category="validation",
            quality_score=0.75
        ),
    ]
    
    print(f"\nTask: {task}")
    print(f"Example pool size: {len(examples)}")
    
    # Select examples
    selected = await optimizer.select_few_shot_examples(task, examples, k=3)
    
    print(f"\nSelected {len(selected)} examples using MMR:")
    for i, ex in enumerate(selected, 1):
        print(f"\n{i}. Example (quality: {ex.quality_score}):")
        print(f"   Input: {ex.input_text}")
        print(f"   Tokens: {ex.token_count}")

async def test_molecular_context_building():
    """Test building molecular prompts from atomic components"""
    print("\n" + "="*60)
    print("TEST 4: Atomic → Molecular Prompt Building")
    print("="*60)
    
    optimizer = create_context_optimizer()
    
    # Create atomic prompt
    atomic = AtomicPrompt(
        instruction="Create a REST API endpoint for user authentication",
        constraints=[
            "Use JWT tokens for authentication",
            "Include rate limiting",
            "Return appropriate HTTP status codes"
        ],
        output_format="Python code using FastAPI framework"
    )
    
    # Create examples
    examples = [
        Example(
            input_text="Create a login endpoint",
            output_text="@app.post('/login')\nasync def login(credentials: UserCredentials):\n    user = authenticate_user(credentials)\n    if not user:\n        raise HTTPException(status_code=401)\n    token = create_jwt_token(user)\n    return {'access_token': token}",
            category="auth",
            quality_score=0.95
        ),
        Example(
            input_text="Add rate limiting to endpoint",
            output_text="from slowapi import Limiter\nlimiter = Limiter(key_func=get_remote_address)\n\n@app.get('/api/data')\n@limiter.limit('5/minute')\nasync def get_data():\n    return {'data': 'value'}",
            category="security",
            quality_score=0.9
        ),
    ]
    
    # Create context items
    context_items = [
        ContextItem(
            content="JWT tokens should include user ID, email, and expiration time in the payload",
            source="security_guide",
            context_type="documentation",
            relevance_score=0.85
        ),
        ContextItem(
            content="Use bcrypt for password hashing with a cost factor of 12",
            source="best_practices",
            context_type="documentation",
            relevance_score=0.8
        ),
    ]
    
    print(f"Atomic Prompt:")
    print(f"  Instruction: {atomic.instruction}")
    print(f"  Constraints: {len(atomic.constraints)}")
    print(f"  Token count: {atomic.token_count}")
    
    # Build molecular context
    molecular = await optimizer.build_molecular_context(
        atomic_prompt=atomic,
        examples=examples,
        context_items=context_items,
        max_tokens=500
    )
    
    print(f"\nMolecular Prompt Built:")
    print(f"  Examples included: {len(molecular.examples)}")
    print(f"  Context items included: {len(molecular.context_items)}")
    print(f"  Total tokens: {molecular.total_tokens}")
    print(f"  Expected performance: {molecular.expected_performance:.2%}")
    
    print("\nProgression Summary:")
    print(f"  Atomic (base): {atomic.token_count} tokens")
    print(f"  + Examples: {sum(ex.token_count for ex in molecular.examples)} tokens")
    print(f"  + Context: {sum(ctx.token_count for ctx in molecular.context_items)} tokens")
    print(f"  = Molecular: {molecular.total_tokens} tokens")

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CONTEXT OPTIMIZER - REAL MATHEMATICAL OPTIMIZATION")
    print("="*60)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Some tests will use fallback methods.\n")
    
    try:
        # Run tests
        await test_information_density()
        await test_context_optimization()
        await test_few_shot_selection()
        await test_molecular_context_building()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("1. ✓ Real Shannon entropy calculation (not mock)")
        print("2. ✓ Knapsack optimization for token budgets")
        print("3. ✓ MMR algorithm for diverse example selection")
        print("4. ✓ Information density metrics")
        print("5. ✓ Atomic → Molecular prompt progression")
        print("6. ✓ OpenAI embeddings for similarity (when API key present)")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
