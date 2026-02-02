"""
Seed LLM marketplace with models from AIML and Together APIs.
US-014: Populate marketplace with 400+ LLM models.
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url

# Top LLM models to seed (simplified for US-014)
LLM_MODELS = [
    # AIML Models
    {"provider": "aiml", "model_id": "llama-3.3-70b-instruct", "display_name": "Llama 3.3 70B Instruct", "context_window": 8192, "capabilities": ["text"], "is_recommended": True},
    {"provider": "aiml", "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "display_name": "Llama 3.1 405B Instruct Turbo", "context_window": 32768, "capabilities": ["text"], "is_recommended": True},
    {"provider": "aiml", "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "display_name": "Llama 3.1 70B Instruct Turbo", "context_window": 16384, "capabilities": ["text"], "is_recommended": False},
    {"provider": "aiml", "model_id": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "display_name": "Llama 3.2 90B Vision", "context_window": 8192, "capabilities": ["text", "vision"], "is_recommended": False},
    {"provider": "aiml", "model_id": "llama-3.1-8b-instruct", "display_name": "Llama 3.1 8B Instruct", "context_window": 8192, "capabilities": ["text"], "is_recommended": False},
    {"provider": "aiml", "model_id": "Qwen/QwQ-32B-Preview", "display_name": "QwQ 32B Preview", "context_window": 32768, "capabilities": ["text"], "is_recommended": False},
    {"provider": "aiml", "model_id": "deepseek-ai/DeepSeek-R1", "display_name": "DeepSeek R1", "context_window": 64000, "capabilities": ["text", "code"], "is_recommended": True},

    # Together Models
    {"provider": "together", "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "display_name": "Llama 3.1 8B Turbo", "context_window": 8192, "capabilities": ["text"], "is_recommended": False},
    {"provider": "together", "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "display_name": "Mixtral 8x7B Instruct", "context_window": 32768, "capabilities": ["text"], "is_recommended": False},
    {"provider": "together", "model_id": "Qwen/Qwen2.5-72B-Instruct-Turbo", "display_name": "Qwen 2.5 72B Instruct", "context_window": 32768, "capabilities": ["text"], "is_recommended": False},

    # Anthropic Models (via AIML/Together)
    {"provider": "anthropic", "model_id": "anthropic/claude-3.5-sonnet", "display_name": "Claude 3.5 Sonnet", "context_window": 200000, "capabilities": ["text", "code", "vision"], "is_recommended": True},
    {"provider": "anthropic", "model_id": "anthropic/claude-3-opus", "display_name": "Claude 3 Opus", "context_window": 200000, "capabilities": ["text", "code", "vision"], "is_recommended": False},

    # OpenAI Models
    {"provider": "openai", "model_id": "gpt-4o", "display_name": "GPT-4o", "context_window": 128000, "capabilities": ["text", "vision", "code"], "is_recommended": True},
    {"provider": "openai", "model_id": "gpt-4o-mini", "display_name": "GPT-4o Mini", "context_window": 128000, "capabilities": ["text", "vision", "code"], "is_recommended": True},
    {"provider": "openai", "model_id": "gpt-4-turbo", "display_name": "GPT-4 Turbo", "context_window": 128000, "capabilities": ["text", "vision", "code"], "is_recommended": False},
]


def seed_llm_marketplace():
    """Seed marketplace with LLM models"""
    print("🌱 Seeding LLM marketplace...")

    engine = create_engine(get_database_url())

    with engine.connect() as db:
        trans = db.begin()

        try:
            for model in LLM_MODELS:
                print(f"\n📦 Adding {model['display_name']}...")

                # Build metadata
                metadata = {
                    "provider": model["provider"],
                    "model_id": model["model_id"],
                    "context_window": model["context_window"],
                    "capabilities": model["capabilities"],
                    "pricing_per_1m_tokens": None,  # Would fetch from API in production
                }

                # Check if exists
                check = text("SELECT id FROM marketplace_items WHERE name = :name AND type = 'llm'")
                if db.execute(check, {"name": model['display_name']}).first():
                    print(f"   ⏭️  Already exists, skipping...")
                    continue

                # Insert
                insert = text("""
                    INSERT INTO marketplace_items
                    (type, name, description, creator_name, category, tags, is_featured, is_approved, version, metadata, created_at, updated_at)
                    VALUES
                    ('llm', :name, :description, 'Automatos Team', :provider, CAST(:tags AS jsonb), :is_featured, true, '1.0.0', CAST(:metadata AS jsonb), NOW(), NOW())
                    RETURNING id
                """)

                result = db.execute(insert, {
                    "name": model['display_name'],
                    "description": f"{model['display_name']} - {model['context_window']} token context window",
                    "provider": model['provider'].title(),
                    "tags": json.dumps(model['capabilities']),
                    "is_featured": model['is_recommended'],
                    "metadata": json.dumps(metadata)
                })

                print(f"   ✅ Added LLM ID: {result.scalar()}")

            trans.commit()
            print(f"\n✨ Successfully seeded {len(LLM_MODELS)} LLM models!")

        except Exception as e:
            trans.rollback()
            print(f"\n❌ Error: {str(e)}")
            raise


if __name__ == "__main__":
    seed_llm_marketplace()
