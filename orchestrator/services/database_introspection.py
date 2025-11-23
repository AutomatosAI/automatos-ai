import logging
import time
from typing import Dict, Any, List
from uuid import UUID
from decimal import Decimal
from datetime import datetime, date

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def make_json_serializable(obj):
    """Convert non-JSON-serializable types to JSON-serializable formats"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


class DatabaseIntrospectionService:
    """
    PRD-21: Schema Introspection (MVP)
    - Supports Postgres and MySQL via SQLAlchemy URLs constructed from credential dict
    - Returns schema metadata JSON-serializable dict
    """

    def __init__(self, credential: Dict[str, Any], dialect: str = "postgres"):
        self.credential = credential
        self.dialect = dialect
        self.engine = self._create_engine()

    def _create_engine(self):
        host = self.credential.get("host")
        port = str(self.credential.get("port"))
        database = self.credential.get("database")
        user = self.credential.get("user")
        password = self.credential.get("password")

        if self.dialect == "postgres":
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        elif self.dialect == "mysql":
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported dialect: {self.dialect}")

        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)

    def introspect(self, include_samples: bool = True, sample_limit: int = 5) -> Dict[str, Any]:
        started = time.time()
        metadata: Dict[str, Any] = {
            "database": {"engine": self.dialect},
            "tables": [],
            "relationships": [],
            "stats": {}
        }

        with self.engine.connect() as conn:
            if self.dialect == "postgres":
                tables = conn.execute(text(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_schema NOT IN ('pg_catalog','information_schema')
                    ORDER BY table_schema, table_name
                    """
                )).fetchall()

                for schema, table in tables:
                    # Columns
                    cols = conn.execute(text(
                        """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = :schema AND table_name = :table
                        ORDER BY ordinal_position
                        """
                    ), {"schema": schema, "table": table}).fetchall()

                    columns: List[Dict[str, Any]] = []
                    for col_name, data_type, is_nullable, column_default in cols:
                        col: Dict[str, Any] = {
                            "name": col_name,
                            "type": data_type,
                            "nullable": (is_nullable == "YES"),
                            "default": column_default
                        }
                        if include_samples:
                            data_type_lower = (data_type or "").lower()
                            if "json" in data_type_lower or "array" in data_type_lower:
                                col["samples"] = []
                                columns.append(col)
                                continue
                            try:
                                sample_sql = text(f'SELECT DISTINCT "{col_name}" FROM "{schema}"."{table}" WHERE "{col_name}" IS NOT NULL LIMIT :lim')
                                samples = conn.execute(sample_sql, {"lim": sample_limit}).fetchall()
                                col["samples"] = [r[0] for r in samples]
                            except SQLAlchemyError:
                                logger.warning(
                                    f"Failed to sample column {col_name} in {table}",
                                    exc_info=True
                                )
                                try:
                                    conn.rollback()
                                except SQLAlchemyError:
                                    logger.debug("Rollback failed or unnecessary", exc_info=True)
                    # Row count
                    try:
                        cnt = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table}"'))
                        row_count = cnt.scalar() or 0
                    except SQLAlchemyError:
                        logger.warning(
                            f"Failed to count rows in {schema}.{table}",
                            exc_info=True
                        )
                        try:
                            conn.rollback()
                        except SQLAlchemyError:
                            logger.debug("Rollback failed or unnecessary", exc_info=True)
                        row_count = 0

                    metadata["tables"].append({
                        "name": table,
                        "schema": schema,
                        "row_count": row_count,
                        "columns": columns
                    })

                # Foreign keys
                fks = conn.execute(text(
                    """
                    SELECT tc.table_schema, tc.table_name, kcu.column_name,
                           ccu.table_schema AS foreign_table_schema,
                           ccu.table_name AS foreign_table_name,
                           ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    """
                )).fetchall()

                for row in fks:
                    (schema, table, column, f_schema, f_table, f_column) = row
                    metadata["relationships"].append({
                        "from_table": table,
                        "from_column": column,
                        "to_table": f_table,
                        "to_column": f_column,
                        "type": "many_to_one"
                    })

            elif self.dialect == "mysql":
                # Similar MySQL queries (using information_schema)
                tables = conn.execute(text(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                    ORDER BY table_name
                    """
                )).fetchall()

                for schema, table in tables:
                    cols = conn.execute(text(
                        """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE() AND table_name = :table
                        ORDER BY ordinal_position
                        """
                    ), {"table": table}).fetchall()

                    columns: List[Dict[str, Any]] = []
                    for col_name, data_type, is_nullable, column_default in cols:
                        col: Dict[str, Any] = {
                            "name": col_name,
                            "type": data_type,
                            "nullable": (is_nullable == "YES"),
                            "default": column_default
                        }
                        if include_samples:
                            data_type_lower = (data_type or "").lower()
                            if "json" in data_type_lower or "array" in data_type_lower:
                                col["samples"] = []
                                columns.append(col)
                                continue
                            try:
                                sample_sql = text(f"SELECT DISTINCT `{col_name}` FROM `{table}` WHERE `{col_name}` IS NOT NULL LIMIT :lim")
                                samples = conn.execute(sample_sql, {"lim": sample_limit}).fetchall()
                                col["samples"] = [r[0] for r in samples]
                            except SQLAlchemyError:
                                try:
                                    conn.rollback()
                                except SQLAlchemyError:
                                    pass
                                col["samples"] = []
                        columns.append(col)

                    try:
                        cnt = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                        row_count = cnt.scalar() or 0
                    except SQLAlchemyError:
                        try:
                            conn.rollback()
                        except SQLAlchemyError:
                            pass
                        row_count = 0

                    metadata["tables"].append({
                        "name": table,
                        "schema": schema,
                        "row_count": row_count,
                        "columns": columns
                    })

                fks = conn.execute(text(
                    """
                    SELECT k.table_name, k.column_name, k.referenced_table_name, k.referenced_column_name
                    FROM information_schema.key_column_usage k
                    WHERE k.referenced_table_name IS NOT NULL AND k.table_schema = DATABASE()
                    """
                )).fetchall()

                for table, column, f_table, f_column in fks:
                    metadata["relationships"].append({
                        "from_table": table,
                        "from_column": column,
                        "to_table": f_table,
                        "to_column": f_column,
                        "type": "many_to_one"
                    })

        metadata["stats"]["introspection_ms"] = int((time.time() - started) * 1000)
        return make_json_serializable(metadata)
