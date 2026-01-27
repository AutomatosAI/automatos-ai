import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from core.database.database import get_db_session
from sqlalchemy import text

def fix_alembic():
    print("Attempting to reset alembic_version table...")
    try:
        with get_db_session() as session:
            # Check current version
            result = session.execute(text("SELECT * FROM alembic_version"))
            rows = result.fetchall()
            print(f"Current versions: {rows}")
            
            # Delete all versions
            session.execute(text("DELETE FROM alembic_version"))
            session.commit()
            print("✅ Cleared alembic_version table.")
            
            # Optionally insert the target version or let alembic stamp handle it
            # Inserting '20260125_composio_metadata'
            session.execute(text("INSERT INTO alembic_version (version_num) VALUES ('20260125_composio_metadata')"))
            session.commit()
            print("✅ Inserted '20260125_composio_metadata' into alembic_version.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_alembic()
