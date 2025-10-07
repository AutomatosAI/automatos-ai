#!/usr/bin/env python3
"""
Seed LLM Models Table
=====================
Seeds the llm_models table with default OpenAI and Anthropic models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from database.database import get_db_session
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_models():
    """Seed the llm_models table with default models"""
    
    with get_db_session() as db:
        # Check if models already exist
        result = db.execute(text("SELECT COUNT(*) FROM llm_models"))
        count = result.scalar()
        
        if count > 0:
            logger.info(f"✅ llm_models table already has {count} models")
            return
        
        logger.info("🌱 Seeding llm_models table...")
        
        # OpenAI Models
        db.execute(text("""
            INSERT INTO llm_models (
                provider, model_id, display_name, model_family,
                capabilities, context_window, max_output_tokens,
                input_cost_per_1k_tokens, output_cost_per_1k_tokens,
                supports_functions, supports_vision, supports_streaming,
                recommended_for, status, description
            ) VALUES
            -- GPT-4 Turbo
            ('openai', 'gpt-4-turbo-preview', 'GPT-4 Turbo', 'gpt-4',
             '{"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "speed": "fast"}',
             128000, 4096, 0.01, 0.03, true, false, true,
             '["code_analysis", "complex_reasoning", "system_design", "architecture"]',
             'active', 'Latest GPT-4 model with improved performance and 128K context window'),
            
            -- GPT-4
            ('openai', 'gpt-4', 'GPT-4', 'gpt-4',
             '{"reasoning": "excellent", "coding": "excellent", "analysis": "excellent"}',
             8192, 4096, 0.03, 0.06, true, false, true,
             '["code_review", "security_audit", "architecture", "complex_tasks"]',
             'active', 'Most capable GPT-4 model for complex reasoning and coding tasks'),
            
            -- GPT-3.5 Turbo
            ('openai', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 'gpt-3.5',
             '{"reasoning": "good", "coding": "good", "speed": "very fast", "cost": "low"}',
             16385, 4096, 0.0005, 0.0015, true, false, true,
             '["simple_tasks", "data_processing", "quick_responses", "high_volume"]',
             'active', 'Fast and cost-effective model for simpler tasks and high-volume operations')
        """))
        
        # Anthropic Models
        db.execute(text("""
            INSERT INTO llm_models (
                provider, model_id, display_name, model_family,
                capabilities, context_window, max_output_tokens,
                input_cost_per_1k_tokens, output_cost_per_1k_tokens,
                supports_functions, supports_vision, supports_streaming,
                recommended_for, status, description
            ) VALUES
            -- Claude 3 Opus
            ('anthropic', 'claude-3-opus-20240229', 'Claude 3 Opus', 'claude-3',
             '{"reasoning": "excellent", "analysis": "excellent", "creativity": "excellent", "context": "very large"}',
             200000, 4096, 0.015, 0.075, false, false, true,
             '["complex_analysis", "research", "planning", "creative_writing"]',
             'active', 'Most powerful Claude model with superior reasoning and 200K context window'),
            
            -- Claude 3 Sonnet
            ('anthropic', 'claude-3-sonnet-20240229', 'Claude 3 Sonnet', 'claude-3',
             '{"reasoning": "excellent", "balance": "optimal", "speed": "fast", "cost": "moderate"}',
             200000, 4096, 0.003, 0.015, false, false, true,
             '["balanced_tasks", "general_purpose", "workflows", "agent_coordination"]',
             'active', 'Balanced model offering excellent performance at moderate cost'),
            
            -- Claude 3 Haiku
            ('anthropic', 'claude-3-haiku-20240307', 'Claude 3 Haiku', 'claude-3',
             '{"speed": "fastest", "cost": "lowest", "reasoning": "good", "efficiency": "excellent"}',
             200000, 4096, 0.00025, 0.00125, false, false, true,
             '["high_volume", "simple_tasks", "cost_sensitive", "real_time"]',
             'active', 'Fastest and most cost-effective Claude model for high-volume operations')
        """))
        
        db.commit()
        
        # Verify
        result = db.execute(text("SELECT COUNT(*) FROM llm_models"))
        count = result.scalar()
        
        logger.info(f"✅ Successfully seeded {count} models into llm_models table!")
        
        # List the models
        result = db.execute(text("""
            SELECT provider, model_id, display_name 
            FROM llm_models 
            ORDER BY provider, model_id
        """))
        
        logger.info("\n📋 Available models:")
        for row in result:
            logger.info(f"   • {row[0]}: {row[1]} ({row[2]})")


if __name__ == "__main__":
    try:
        seed_models()
        logger.info("\n✨ Model seeding completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error seeding models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

