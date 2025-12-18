#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/orchestrator')

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database connection
DATABASE_URL = "postgresql://postgres:secure_password_123@127.0.0.1:5432/orchestrator_db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

try:
    # Check skills with NULL categories
    result = session.execute(text("""
        SELECT 
            COUNT(*) as total_skills,
            COUNT(category) as skills_with_category,
            COUNT(*) - COUNT(category) as null_categories
        FROM skills
    """))
    row = result.fetchone()
    print(f"Total skills: {row[0]}")
    print(f"Skills with category: {row[1]}")
    print(f"Skills with NULL category: {row[2]}")
    
    # Check if there are any agents
    result = session.execute(text("SELECT COUNT(*) FROM agents"))
    agent_count = result.fetchone()[0]
    print(f"\nTotal agents: {agent_count}")
    
    # Check for skills with NULL category
    if row[2] > 0:
        print(f"\n⚠️  Found {row[2]} skills with NULL category - this may cause API errors")
        result = session.execute(text("""
            SELECT id, name, skill_type, category 
            FROM skills 
            WHERE category IS NULL 
            LIMIT 5
        """))
        print("\nSample skills with NULL category:")
        for skill in result:
            print(f"  ID: {skill[0]}, Name: {skill[1]}, Type: {skill[2]}, Category: {skill[3]}")
    
finally:
    session.close()
