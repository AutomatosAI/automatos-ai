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
from core.database.database import get_db_session
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_models():
    """Seed the llm_models table with default models"""
    
    with get_db_session() as db:
        # Check current model count
        result = db.execute(text("SELECT COUNT(*) FROM llm_models"))
        count = result.scalar()
        
        logger.info(f"📊 Current models in database: {count}")
        logger.info("🌱 Seeding/updating llm_models table...")
        
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
            ON CONFLICT (model_id) DO NOTHING
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
            ON CONFLICT (model_id) DO NOTHING
        """))
        
        # Google Gemini Models
        db.execute(text("""
            INSERT INTO llm_models (
                provider, model_id, display_name, model_family,
                capabilities, context_window, max_output_tokens,
                input_cost_per_1k_tokens, output_cost_per_1k_tokens,
                supports_functions, supports_vision, supports_streaming,
                recommended_for, status, description
            ) VALUES
            -- Gemini 2.5 Pro
            ('google', 'gemini-2.5-pro', 'Gemini 2.5 Pro', 'gemini-2.5',
             '{"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "multimodal": "advanced"}',
             1000000, 8192, 0.00125, 0.005, true, true, true,
             '["complex_reasoning", "code_security", "multimodal_analysis", "long_context"]',
             'active', 'Google''s most intelligent AI model with 1M token context window'),
            
            -- Gemini 2.5 Flash
            ('google', 'gemini-2.5-flash', 'Gemini 2.5 Flash', 'gemini-2.5',
             '{"speed": "very fast", "reasoning": "excellent", "efficiency": "excellent", "cost": "low"}',
             1000000, 8192, 0.000075, 0.0003, true, true, true,
             '["high_volume", "fast_processing", "real_time", "cost_efficient"]',
             'active', 'Fast and efficient model with excellent performance at low cost'),
            
            -- Gemini 1.5 Pro
            ('google', 'gemini-1.5-pro', 'Gemini 1.5 Pro', 'gemini-1.5',
             '{"reasoning": "excellent", "context": "very large", "analysis": "excellent"}',
             2000000, 8192, 0.00125, 0.005, true, true, true,
             '["long_document_analysis", "complex_tasks", "research", "code_analysis"]',
             'active', 'Previous generation Gemini with exceptional 2M token context window')
            ON CONFLICT (model_id) DO NOTHING
        """))
        
        # AWS Bedrock Models (Cost-effective gateway to multiple providers)
        db.execute(text("""
            INSERT INTO llm_models (
                provider, model_id, display_name, model_family,
                capabilities, context_window, max_output_tokens,
                input_cost_per_1k_tokens, output_cost_per_1k_tokens,
                supports_functions, supports_vision, supports_streaming,
                recommended_for, status, description
            ) VALUES
            -- Claude 3.5 Sonnet (via Bedrock)
            ('aws_bedrock', 'anthropic.claude-3-5-sonnet-20241022-v2:0', 'Claude 3.5 Sonnet (Bedrock)', 'claude-3',
             '{"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "cost": "low"}',
             200000, 8192, 0.003, 0.015, true, false, true,
             '["cost_efficient", "complex_reasoning", "code_analysis", "balanced_tasks"]',
             'active', 'Claude 3.5 Sonnet via AWS Bedrock - 80% cost savings vs direct API'),
            
            -- Claude 3 Haiku (via Bedrock)
            ('aws_bedrock', 'anthropic.claude-3-haiku-20240307-v1:0', 'Claude 3 Haiku (Bedrock)', 'claude-3',
             '{"speed": "fastest", "cost": "lowest", "reasoning": "good", "efficiency": "excellent"}',
             200000, 4096, 0.00025, 0.00125, true, false, true,
             '["high_volume", "cost_sensitive", "real_time", "simple_tasks"]',
             'active', 'Fastest Claude model via AWS Bedrock at 75% cost savings'),
            
            -- Meta Llama 3.1 70B (via Bedrock)
            ('aws_bedrock', 'meta.llama3-1-70b-instruct-v1:0', 'Llama 3.1 70B (Bedrock)', 'llama-3',
             '{"reasoning": "excellent", "coding": "good", "cost": "very low", "open_source": "true"}',
             128000, 4096, 0.00099, 0.00099, true, false, true,
             '["cost_sensitive", "open_source", "bulk_processing", "experimentation"]',
             'active', 'Open-source Llama 3.1 70B via AWS Bedrock at extremely low cost'),
            
            -- Meta Llama 3.1 8B (via Bedrock)
            ('aws_bedrock', 'meta.llama3-1-8b-instruct-v1:0', 'Llama 3.1 8B (Bedrock)', 'llama-3',
             '{"speed": "very fast", "cost": "extremely low", "reasoning": "good", "efficiency": "excellent"}',
             128000, 2048, 0.00022, 0.00022, true, false, true,
             '["extreme_cost_savings", "high_volume", "simple_tasks", "experimentation"]',
             'active', 'Smallest Llama model via AWS Bedrock - cheapest option for bulk operations')
            ON CONFLICT (model_id) DO NOTHING
        """))
        
        # HuggingFace Models (Free alternatives for development and experimentation)
        db.execute(text("""
            INSERT INTO llm_models (
                provider, model_id, display_name, model_family,
                capabilities, context_window, max_output_tokens,
                input_cost_per_1k_tokens, output_cost_per_1k_tokens,
                supports_functions, supports_vision, supports_streaming,
                recommended_for, status, description
            ) VALUES
            -- Mistral 7B Instruct
            ('huggingface', 'mistralai/Mistral-7B-Instruct-v0.2', 'Mistral 7B Instruct', 'mistral',
             '{"reasoning": "good", "speed": "fast", "cost": "free", "open_source": "true"}',
             8192, 2048, 0.0, 0.0, false, false, true,
             '["experimentation", "development", "cost_free", "open_source"]',
             'active', 'Free open-source Mistral model via HuggingFace Inference API'),
            
            -- Llama 2 70B Chat
            ('huggingface', 'meta-llama/Llama-2-70b-chat-hf', 'Llama 2 70B Chat', 'llama-2',
             '{"reasoning": "excellent", "cost": "free", "open_source": "true"}',
             4096, 2048, 0.0, 0.0, false, false, true,
             '["free_alternative", "open_source", "experimentation", "development"]',
             'active', 'Free Llama 2 70B model via HuggingFace Inference API'),
            
            -- Zephyr 7B Beta
            ('huggingface', 'HuggingFaceH4/zephyr-7b-beta', 'Zephyr 7B Beta', 'mistral',
             '{"reasoning": "good", "speed": "fast", "cost": "free", "chat": "optimized"}',
             8192, 2048, 0.0, 0.0, false, false, true,
             '["chatbot", "free", "development", "testing"]',
             'active', 'Free fine-tuned chat model optimized for conversations'),
            
            -- Newton Insights Cannabis Extraction Science
            ('huggingface', 'KellanF89/Newton-Insights-V1-cannabis-extraction-science', 'Newton Insights V1', 'custom',
             '{"domain": "cannabis_extraction", "science": "excellent", "specialized": "true"}',
             4096, 2048, 0.0, 0.0, false, false, true,
             '["cannabis_science", "extraction", "specialized_domain", "research"]',
             'active', 'Specialized model for cannabis extraction science and research')
            ON CONFLICT (model_id) DO NOTHING
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

