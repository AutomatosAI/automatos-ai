import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.database.database import get_db_session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


def fix_alembic():
    print("Attempting to reset alembic_version table...")
    try:
        with get_db_session() as session:
            result = session.execute(text("SELECT * FROM alembic_version"))
            rows = result.fetchall()
            print(f"Current versions: {rows}")

            session.execute(text("DELETE FROM alembic_version"))
            session.execute(
                text(
                    "INSERT INTO alembic_version (version_num) VALUES ('20260125_composio_metadata')"
                )
            )
            print("Cleared alembic_version and inserted '20260125_composio_metadata'.")
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    try:
        fix_alembic()
    except Exception:
        sys.exit(1)
