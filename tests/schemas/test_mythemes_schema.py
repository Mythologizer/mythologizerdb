import os
import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.db import (
    session_scope,
    get_table_row_counts,
    clear_all_rows,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


class TestMythemesSchema:
    """Test the mythemes table schema and operations."""
    
    @pytest.mark.integration
    def test_mythemes_table_structure(self):
        """Test that mythemes table has the correct structure."""
        with session_scope() as session:
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
    def test_mythemes_insert_and_count_data(self):
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
    def test_mythemes_vector_operations(self):
        """Test vector operations in the mythemes table."""
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
    def test_mythemes_error_handling(self):
        """Test error handling in mythemes table operations."""
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
    def test_mythemes_concurrent_operations(self):
        """Test concurrent database operations on mythemes table."""
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
