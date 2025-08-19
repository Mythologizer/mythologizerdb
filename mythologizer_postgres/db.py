import os
import re
import logging
from typing import Optional, Mapping, Sequence, Dict, Iterator
from contextlib import contextmanager
from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector


from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import sessionmaker, Session

from . import schema

logger = logging.getLogger(__name__)


class MissingEnvironmentVariable(RuntimeError):
    """Raised when a required environment variable is not set."""


def need(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingEnvironmentVariable(f"{name} is missing")
    return value


def build_url() -> URL:
    return URL.create(
        drivername="postgresql+psycopg",
        username=need("POSTGRES_USER"),
        password=need("POSTGRES_PASSWORD"),
        host=need("POSTGRES_HOST"),
        port=int(need("POSTGRES_PORT")),
        database=need("POSTGRES_DB"),
    )


@lru_cache
def get_engine() -> Engine:
    engine = create_engine(build_url(), pool_pre_ping=True, future=True)

    @event.listens_for(engine, "connect")
    def on_connect(dbapi_conn, _):
        # register pgvector adapter so VECTOR columns come back as Vector objects
        register_vector(dbapi_conn)

    return engine


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def psycopg_connection() -> Iterator[psycopg.Connection]:
    """Get a direct psycopg connection for multi-vector operations."""
    conn = psycopg.connect(
        dbname=need("POSTGRES_DB"),
        user=need("POSTGRES_USER"),
        password=need("POSTGRES_PASSWORD"),
        host=need("POSTGRES_HOST"),
        port=int(need("POSTGRES_PORT")),
    )
    
    # Register vector types for multi-vector support
    register_vector(conn)
    
    try:
        yield conn
    finally:
        conn.close()


def apply_schemas(dim: int) -> None:
    """
    Execute all schema definitions from the schemas folder in order.
    If any execution fails, the error is propagated.
    """
    try:
        # Get all schemas from the schemas folder
        schema_sqls = list(schema.get_schemas("schemas", dim=dim))
        
        if not schema_sqls:
            logger.warning("No schema files found in schemas folder")
            return
        
        # Create a basic engine without vector registration for the initial setup
        # This avoids the chicken-and-egg problem with pgvector extension
        basic_url = build_url()
        basic_engine = create_engine(basic_url)
        
        # Apply the first schema (init.sql.j2) with basic engine
        if schema_sqls:
            with basic_engine.begin() as conn:
                logger.info("Applying initial schema")
                conn.execute(text(schema_sqls[0]))
        
        basic_engine.dispose()
        
        # Now use the regular engine with vector registration for the rest
        engine = get_engine()
        with engine.begin() as conn:  # begin() ensures commit or rollback
            for i, schema_sql in enumerate(schema_sqls[1:], 1):  # Skip first schema, already done
                logger.info("Applying schema %d of %d", i + 1, len(schema_sqls))
                conn.execute(text(schema_sql))

        logger.info("All schemas applied successfully")
    except Exception:
        logger.exception("Error occurred while applying schemas")
        raise


_SCHEMA_NAME_PATTERN = re.compile(
    r"CREATE\s+SCHEMA(?:\s+IF\s+NOT\s+EXISTS)?\s+([a-zA-Z0-9_]+)",
    re.IGNORECASE,
)


def _extract_schema_names(sql_text: str) -> Sequence[str]:
    return _SCHEMA_NAME_PATTERN.findall(sql_text)


def check_if_tables_exist(expected_tables: Sequence[str]) -> Dict[str, bool]:
    """
    Check whether the expected tables exist in the public schema.
    """
    placeholders = ", ".join([f":tbl_{i}" for i in range(len(expected_tables))])
    bind_values = {f"tbl_{i}": name for i, name in enumerate(expected_tables)}

    with get_engine().connect() as conn:
        result = conn.execute(
            text(
                f"""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public' AND tablename IN ({placeholders})
                """
            ),
            bind_values,
        )
        existing = {row[0] for row in result.fetchall()}

    return {name: name in existing for name in expected_tables}



def ping_db(timeout_seconds: float = 5.0) -> bool:
    """
    Perform a lightweight check that the database is reachable.
    """
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            success = bool(row and row[0] == 1)
            if not success:
                logger.warning("Ping query returned unexpected result %s", row)
            return success
    except Exception:
        logger.exception("Database ping failed")
        return False


def clear_all_rows() -> None:
    """
    Delete all rows from all user-defined tables in the 'public' schema.
    """
    try:
        with get_engine().begin() as conn:
            conn.execute(text("""
                DO $$
                DECLARE
                    tbl RECORD;
                BEGIN
                    FOR tbl IN
                        SELECT tablename
                        FROM pg_tables
                        WHERE schemaname = 'public'
                    LOOP
                        EXECUTE format('DELETE FROM %I.%I', 'public', tbl.tablename);
                    END LOOP;
                END $$;
            """))
        logger.info("All rows deleted from public schema")
    except Exception:
        logger.exception("Error occurred while deleting rows")
        raise


def drop_all_tables() -> None:
    """
    Drop all tables in the 'public' schema.
    This will completely remove all table structures and data.
    """
    try:
        with get_engine().begin() as conn:
            # Disable foreign key checks temporarily
            conn.execute(text("SET session_replication_role = replica;"))
            
            # Drop all tables in public schema
            conn.execute(text("""
                DO $$
                DECLARE
                    tbl RECORD;
                BEGIN
                    FOR tbl IN
                        SELECT tablename
                        FROM pg_tables
                        WHERE schemaname = 'public'
                    LOOP
                        EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE', 'public', tbl.tablename);
                    END LOOP;
                END $$;
            """))
            
            # Re-enable foreign key checks
            conn.execute(text("SET session_replication_role = DEFAULT;"))
            
        logger.info("All tables dropped from public schema")
    except Exception:
        logger.exception("Error occurred while dropping tables")
        raise


def get_table_row_counts() -> Dict[str, int]:
    """
    Returns a dictionary mapping table names in the 'public' schema to row counts.
    """
    counts = {}
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]

            for table in tables:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM public.{table}"))
                count = count_result.scalar()
                counts[table] = count
    except Exception:
        logger.exception("Failed to get table row counts")
    return counts

def is_correct_embedding_size(embedding_size: int) -> bool:
    """
    Check if the embedding size matches the vector dimensions in myths and mythemes tables.
    
    Args:
        embedding_size: The embedding dimension to check against
        
    Returns:
        True if the embedding size matches both tables, False otherwise
    """
    try:
        with get_engine().connect() as conn:
            # Simple query to get vector dimensions from both tables
            result = conn.execute(text("""
                SELECT 
                    format_type(a.atttypid, a.atttypmod) as column_type
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public'
                AND c.relname IN ('myths', 'mythemes')
                AND a.attname = 'embedding';
            """))
            
            rows = result.fetchall()
            
            # Check if we found both tables and dimensions match
            if len(rows) != 2:
                logger.warning(f"Expected 2 embedding columns, found {len(rows)}")
                return False
            
            # Verify both tables have the correct vector dimension
            expected_type = f'vector({embedding_size})'
            for row in rows:
                if expected_type not in row[0]:
                    logger.warning(f"Expected {expected_type}, found {row[0]}")
                    return False
            
            return True
            
    except Exception as e:
        logger.exception(f"Error checking embedding size: {e}")
        return False