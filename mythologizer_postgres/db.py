import os
import re
import logging
from typing import Optional, Mapping, Sequence, Dict, Iterator
from contextlib import contextmanager
from functools import lru_cache

import psycopg2
from pgvector.psycopg2 import register_vector


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
        drivername="postgresql+psycopg2",
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


def apply_schemas(dim: int) -> None:
    """
    Execute the schema definitions in order: init, mytheme, myth.
    If any execution fails, the error is propagated.
    """
    schemas = [
        ("init", schema.get_init_schema()),
        ("mytheme", schema.get_mytheme_schema(dim)),
        ("myth", schema.get_myth_schema(dim)),
    ]

    try:
        engine = get_engine()
        with engine.begin() as conn:  # begin() ensures commit or rollback
            for name, schema_sql in schemas:
                logger.info("Applying schema %s", name)
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
        with get_engine().connect() as conn:
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
