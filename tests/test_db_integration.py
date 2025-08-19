import os
import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.db import (
    get_engine,
    session_scope,
    psycopg_connection,
    apply_schemas,
    check_if_tables_exist,
    ping_db,
    get_table_row_counts,
    clear_all_rows,
    is_correct_embedding_size,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable.
    
    This function reads the EMBEDDING_DIM from the environment (typically from .env.test)
    and returns it as an integer. This allows tests to work with different embedding
    dimensions without hardcoding values.
    
    Returns:
        int: The embedding dimension from environment, defaults to 4 if not set
    """
    return int(os.getenv('EMBEDDING_DIM', '4'))


class TestDatabaseIntegration:
    """Integration tests that require a real database connection."""
    
    @pytest.mark.integration
    def test_database_connectivity(self):
        """Test that the database is online and accessible."""
        result = ping_db()
        assert result is True, "Database should be online and accessible"
    
    @pytest.mark.integration
    def test_schema_application_and_table_existence(self):
        """Test that schemas are applied and tables exist."""
        # Check if the expected tables exist (schemas are applied automatically)
        expected_tables = ['myths', 'mythemes', 'agents', 'agent_myths']
        table_existence = check_if_tables_exist(expected_tables)
        
        # All tables should exist if schemas were applied correctly
        assert table_existence['myths'], "myths table should exist"
        assert table_existence['mythemes'], "mythemes table should exist"
        assert table_existence['agents'], "agents table should exist"
        assert table_existence['agent_myths'], "agent_myths table should exist"
    
    @pytest.mark.integration
    def test_insert_and_count_data(self):
        """Test inserting data, counting rows, and clearing data."""
        # Get initial row counts
        initial_counts = get_table_row_counts()
        
        # Insert test data into mythemes table
        with session_scope() as session:
            # Create a test embedding using dimension from environment
            embedding_dim = get_embedding_dim()
            test_embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert test data
            session.execute(text("""
                INSERT INTO mythemes (sentence, embedding) 
                VALUES (:sentence, :embedding)
            """), {
                'sentence': 'Test mythology theme',
                'embedding': test_embedding
            })
            
            # Insert another test record
            session.execute(text("""
                INSERT INTO mythemes (sentence, embedding) 
                VALUES (:sentence, :embedding)
            """), {
                'sentence': 'Another test theme',
                'embedding': np.random.rand(embedding_dim).tolist()
            })
        
        # Check that row counts increased
        after_insert_counts = get_table_row_counts()
        assert after_insert_counts['mythemes'] == initial_counts['mythemes'] + 2, \
            "mythemes table should have 2 more rows after insertion"
        
        # Clear all rows
        clear_all_rows()
        
        # Check that all tables are empty
        empty_counts = get_table_row_counts()
        for table, count in empty_counts.items():
            assert count == 0, f"Table {table} should be empty after clear_all_rows"
    
    @pytest.mark.integration
    def test_psycopg_connection_with_vector_operations(self):
        """Test psycopg connection for vector operations."""
        with psycopg_connection() as conn:
            # Test that we can execute a simple query
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                assert result[0] == 1
            
            # Test that vector extension is available
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                assert result is not None, "pgvector extension should be installed"
    

    

    

    
 