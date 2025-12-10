#!/usr/bin/env python3
"""
Update chunk metadata to include file_path for document downloads
"""

from core.database.database import SessionLocal
from sqlalchemy import text
import json

def update_chunk_metadata():
    """Add file_path to all chunk metadata"""
    db = SessionLocal()
    
    try:
        # Get all chunks with their source_file
        result = db.execute(text("""
            SELECT id, metadata 
            FROM document_chunks
        """))
        chunks = result.fetchall()
        
        print(f"Found {len(chunks)} chunks")
        
        updated = 0
        for chunk_id, metadata in chunks:
            # Metadata is already a dict from JSONB column
            if not isinstance(metadata, dict):
                metadata = json.loads(metadata) if metadata else {}
            
            # Get source file
            source_file = metadata.get('source_file')
            if not source_file:
                continue
            
            # Add file_path if not already present
            if 'file_path' not in metadata:
                metadata['file_path'] = f"/var/automatos/documents/{source_file}"
                
                # Update chunk - use proper parameter binding
                db.execute(text("""
                    UPDATE document_chunks 
                    SET metadata = CAST(:metadata AS jsonb)
                    WHERE id = :chunk_id
                """), {
                    'metadata': json.dumps(metadata),
                    'chunk_id': chunk_id
                })
                updated += 1
                
                if updated % 500 == 0:
                    print(f"  Updated {updated} chunks...")
                    db.commit()  # Commit in batches
        
        db.commit()
        print(f"✅ Updated {updated} chunks with file_path")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_chunk_metadata()
