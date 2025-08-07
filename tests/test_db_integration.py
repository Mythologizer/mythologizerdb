import os
import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.db import (
    ping_db,
    check_if_tables_exist,
    get_table_row_counts,
    clear_all_rows,
    session_scope,
    psycopg_connection,
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
        expected_tables = ['myths', 'mythemes']
        table_existence = check_if_tables_exist(expected_tables)
        
        # Both tables should exist if schemas were applied correctly
        assert table_existence['myths'], "myths table should exist"
        assert table_existence['mythemes'], "mythemes table should exist"
    
    @pytest.mark.integration
    def test_table_structure(self):
        """Test that tables have the correct structure."""
        with session_scope() as session:
            # Check myths table structure
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'myths' 
                ORDER BY ordinal_position
            """))
            myths_columns = {row[0]: row[1] for row in result.fetchall()}
            
            expected_myths_columns = {
                'id': 'integer',
                'embedding': 'USER-DEFINED',  # VECTOR type
                'embedding_ids': 'ARRAY',
                'offsets': 'ARRAY',
                'weights': 'ARRAY'
            }
            
            for col, expected_type in expected_myths_columns.items():
                assert col in myths_columns, f"Column {col} should exist in myths table"
                if expected_type != 'USER-DEFINED':  # Skip VECTOR type check
                    assert myths_columns[col] == expected_type, f"Column {col} should be {expected_type}"
            
            # Check mythemes table structure
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'mythemes' 
                ORDER BY ordinal_position
            """))
            mythemes_columns = {row[0]: row[1] for row in result.fetchall()}
            
            expected_mythemes_columns = {
                'id': 'integer',
                'sentence': 'text',
                'embedding': 'USER-DEFINED'  # VECTOR type
            }
            
            for col, expected_type in expected_mythemes_columns.items():
                assert col in mythemes_columns, f"Column {col} should exist in mythemes table"
                if expected_type != 'USER-DEFINED':  # Skip VECTOR type check
                    assert mythemes_columns[col] == expected_type, f"Column {col} should be {expected_type}"
    
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
    
    @pytest.mark.integration
    def test_vector_operations(self):
        """Test vector operations in the database."""
        with session_scope() as session:
            # Create test embeddings using dimension from environment
            embedding_dim = get_embedding_dim()
            embedding1 = np.random.rand(embedding_dim).tolist()
            embedding2 = np.random.rand(embedding_dim).tolist()
            
            # Insert test data with vectors
            session.execute(text("""
                INSERT INTO mythemes (sentence, embedding) 
                VALUES (:sentence, :embedding)
            """), {
                'sentence': 'Vector test theme 1',
                'embedding': embedding1
            })
            
            session.execute(text("""
                INSERT INTO mythemes (sentence, embedding) 
                VALUES (:sentence, :embedding)
            """), {
                'sentence': 'Vector test theme 2',
                'embedding': embedding2
            })
            
            # Test vector similarity query
            result = session.execute(text("""
                SELECT sentence, embedding <-> (:query_embedding)::vector as distance
                FROM mythemes
                ORDER BY embedding <-> (:query_embedding)::vector
                LIMIT 1
            """), {
                'query_embedding': embedding1
            })
            
            row = result.fetchone()
            assert row is not None, "Should find at least one result"
            assert 'Vector test theme 1' in row[0], "Should find the exact match first"
            assert row[1] == 0.0, "Distance to self should be 0"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_myths_table_complex_operations(self):
        """Test the myths table with complex vector operations."""
        with session_scope() as session:
            # Create test data for myths table using dimension from environment
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            embedding_ids = [1, 2, 3]
            
            # Insert test data (simplified - just test the basic embedding and embedding_ids)
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, :embedding_ids, ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding,
                'embedding_ids': embedding_ids
            })
            
            # Verify the data was inserted correctly
            result = session.execute(text("SELECT embedding, embedding_ids FROM myths"))
            row = result.fetchone()
            assert row is not None, "Should find the inserted row"
            assert len(row[1]) == 3, "embedding_ids should have 3 elements"  # row[1] is embedding_ids
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_error_handling(self):
        """Test error handling in database operations."""
        # Test with invalid vector dimension
        with session_scope() as session:
            with pytest.raises(Exception):
                # Try to insert a vector with wrong dimension
                embedding_dim = get_embedding_dim()
                wrong_dimension = embedding_dim - 1  # Use wrong dimension
                session.execute(text("""
                    INSERT INTO mythemes (sentence, embedding) 
                    VALUES (:sentence, :embedding)
                """), {
                    'sentence': 'Invalid embedding test',
                    'embedding': [1.0] * wrong_dimension  # Wrong dimension
                })
    
    @pytest.mark.integration
    def test_concurrent_operations(self):
        """Test concurrent database operations."""
        import threading
        import time
        
        results = []
        
        def insert_data(thread_id):
            try:
                with session_scope() as session:
                    embedding_dim = get_embedding_dim()
                    embedding = np.random.rand(embedding_dim).tolist()  # Use dimension from environment
                    session.execute(text("""
                        INSERT INTO mythemes (sentence, embedding)
                        VALUES (:sentence, :embedding)
                    """), {
                        'sentence': f'Thread {thread_id} data',
                        'embedding': embedding
                    })
                    results.append(f"Thread {thread_id} success")
            except Exception as e:
                results.append(f"Thread {thread_id} failed: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=insert_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all threads succeeded
        assert len(results) == 3, "All threads should complete"
        assert all("success" in result for result in results), "All threads should succeed"
        
        # Verify data was inserted
        counts = get_table_row_counts()
        assert counts['mythemes'] >= 3, "Should have at least 3 rows from concurrent inserts"
        
        # Clean up
        clear_all_rows() 