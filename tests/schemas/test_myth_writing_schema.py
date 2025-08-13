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


class TestMythWritingsSchema:
    """Test the myth_writings table schema and operations."""
    
    @pytest.mark.integration
    def test_myth_writings_table_structure(self):
        """Test that myth_writings table has the correct structure."""
        with session_scope() as session:
            # Check myth_writings table structure
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'myth_writings' 
                ORDER BY ordinal_position
            """))
            myth_writings_columns = {row[0]: row[1] for row in result.fetchall()}
            
            expected_myth_writings_columns = {
                'id': 'integer',
                'myth_id': 'integer',
                'written_text': 'text',
                'created_at': 'timestamp without time zone'
            }
            
            for col, expected_type in expected_myth_writings_columns.items():
                assert col in myth_writings_columns, f"Column {col} should exist in myth_writings table"
                assert myth_writings_columns[col] == expected_type, f"Column {col} should be {expected_type}"
    
    @pytest.mark.integration
    def test_myth_writings_basic_operations(self):
        """Test basic CRUD operations on myth_writings table."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert a myth
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            # Get the myth ID
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            # Insert myth writings
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': 'This is the first version of the myth about Zeus and his adventures.'
            })
            
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': 'This is the second version with more details about Zeus and his family.'
            })
            
            # Query myth writings
            result = session.execute(text("""
                SELECT id, myth_id, written_text, created_at 
                FROM myth_writings 
                ORDER BY created_at
            """))
            writings = result.fetchall()
            
            assert len(writings) == 2, "Should have 2 myth writings"
            assert writings[0][1] == myth_id, "First writing should belong to the same myth"
            assert writings[1][1] == myth_id, "Second writing should belong to the same myth"
            assert 'first version' in writings[0][2], "First writing should contain 'first version'"
            assert 'second version' in writings[1][2], "Second writing should contain 'second version'"
            
            # Update myth writing
            session.execute(text("""
                UPDATE myth_writings 
                SET written_text = :new_text 
                WHERE id = :writing_id
            """), {
                'new_text': 'Updated version of the myth with corrections.',
                'writing_id': writings[0][0]
            })
            
            # Verify update
            result = session.execute(text("""
                SELECT written_text FROM myth_writings WHERE id = :writing_id
            """), {
                'writing_id': writings[0][0]
            })
            updated_text = result.fetchone()[0]
            assert 'Updated' in updated_text, "Text should be updated"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_myth_writings_constraints(self):
        """Test that myth_writings table constraints work correctly."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert a myth
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            # Get the myth ID
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            # Test valid insertion
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': 'Valid myth writing text'
            })
            
            # Test NOT NULL constraints
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO myth_writings (myth_id, written_text) 
                    VALUES (:myth_id, :written_text)
                """), {
                    'myth_id': None,  # Should fail
                    'written_text': 'Valid text'
                })
            
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO myth_writings (myth_id, written_text) 
                    VALUES (:myth_id, :written_text)
                """), {
                    'myth_id': myth_id,
                    'written_text': None  # Should fail
                })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_myth_writings_foreign_key_constraints(self):
        """Test that myth_writings foreign key constraints work correctly."""
        with session_scope() as session:
            # Try to insert with non-existent myth_id
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO myth_writings (myth_id, written_text) 
                    VALUES (:myth_id, :written_text)
                """), {
                    'myth_id': 999,  # Non-existent myth
                    'written_text': 'Valid text'
                })
    
    @pytest.mark.integration
    def test_myth_writings_cascade_delete(self):
        """Test that myth_writings records are deleted when parent myth is deleted."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert a myth
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            # Get the myth ID
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            # Insert myth writings
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': 'First version of the myth'
            })
            
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': 'Second version of the myth'
            })
            
            # Verify records exist
            result = session.execute(text("SELECT COUNT(*) FROM myth_writings"))
            count_before = result.fetchone()[0]
            assert count_before == 2, "Should have 2 myth_writings records"
            
            # Delete the myth
            session.execute(text("DELETE FROM myths WHERE id = :myth_id"), {
                'myth_id': myth_id
            })
            
            # Verify myth_writings records were deleted
            result = session.execute(text("SELECT COUNT(*) FROM myth_writings"))
            count_after = result.fetchone()[0]
            assert count_after == 0, "myth_writings records should be deleted when myth is deleted"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_myth_writings_timestamps(self):
        """Test that created_at timestamps work correctly."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert a myth
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            # Get the myth ID
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            # Insert a myth writing
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': 'Test myth writing'
            })
            
            # Check that timestamp was set
            result = session.execute(text("""
                SELECT created_at 
                FROM myth_writings 
                WHERE myth_id = :myth_id
            """), {
                'myth_id': myth_id
            })
            created_at = result.fetchone()[0]
            assert created_at is not None, "created_at should be set"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_myth_writings_multiple_myths(self):
        """Test myth_writings with multiple myths."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding1 = np.random.rand(embedding_dim).tolist()
            embedding2 = np.random.rand(embedding_dim).tolist()
            
            # Insert two myths
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding1
            })
            
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[2], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding2
            })
            
            # Get the myth IDs
            myth_result = session.execute(text("SELECT id FROM myths ORDER BY id"))
            myth_ids = [row[0] for row in myth_result.fetchall()]
            
            # Insert myth writings for both myths
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_ids[0],
                'written_text': 'Writing for first myth'
            })
            
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_ids[1],
                'written_text': 'Writing for second myth'
            })
            
            # Verify each myth has its own writings
            result = session.execute(text("""
                SELECT myth_id, COUNT(*) as writing_count 
                FROM myth_writings 
                GROUP BY myth_id 
                ORDER BY myth_id
            """))
            writing_counts = result.fetchall()
            
            assert len(writing_counts) == 2, "Should have writings for 2 myths"
            assert writing_counts[0][1] == 1, "First myth should have 1 writing"
            assert writing_counts[1][1] == 1, "Second myth should have 1 writing"
        
        # Clean up
        clear_all_rows()
