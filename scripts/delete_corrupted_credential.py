#!/usr/bin/env python3
"""Delete corrupted HuggingFace credential"""
import sys
sys.path.insert(0, '/root/automatos-ai/orchestrator')

from database.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Delete audit logs first (credential_id is VARCHAR)
    db.execute(text("DELETE FROM credential_audit_logs WHERE credential_id = CAST(:id AS VARCHAR)"), {'id': 15})
    # Delete the credential
    db.execute(text('DELETE FROM credentials WHERE id = :id'), {'id': 15})
    db.commit()
    print('✅ Deleted corrupted credential ID 15 (development_huggingface)')
    print('You can now recreate it via the UI with your HuggingFace API token.')
except Exception as e:
    db.rollback()
    print(f'❌ Error: {e}')
finally:
    db.close()

