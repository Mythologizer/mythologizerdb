import pytest
import os
from sqlalchemy import text

from mythologizer_postgres.db import (
    get_engine,
    apply_schemas,
    check_if_tables_exist,
    drop_all_tables,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


class TestDropAllTables:
    """Test the drop_all_tables function in isolation."""
    
    @pytest.mark.integration
    def test_drop_all_tables_functionality(self):
        """Test that drop_all_tables removes all tables from the database."""
        # The Makefile already creates tables, so we don't need to apply schemas again
        # Just verify tables exist first
        expected_tables = ['myths', 'mythemes', 'agents', 'agent_myths', 'agent_attribute_defs']
        table_existence = check_if_tables_exist(expected_tables)
        
        # Verify tables exist (created by Makefile)
        for table in expected_tables:
            assert table_existence[table], f"Table {table} should exist (created by Makefile)"
        
        # Drop all tables
        drop_all_tables()
        
        # Verify all tables are gone
        table_existence_after = check_if_tables_exist(expected_tables)
        for table in expected_tables:
            assert not table_existence_after[table], f"Table {table} should not exist after drop_all_tables"
        
        # Verify no tables exist in public schema
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """))
            remaining_tables = [row[0] for row in result.fetchall()]
            
        assert len(remaining_tables) == 0, f"Expected no tables in public schema, found: {remaining_tables}"
