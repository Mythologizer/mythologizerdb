import pytest
import asyncio
import json
import psycopg
import numpy as np
from sqlalchemy import text
from unittest.mock import patch

from mythologizer_postgres.db import (
    session_scope,
    build_url,
    get_table_row_counts,
    clear_all_rows,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    import os
    return int(os.getenv('EMBEDDING_DIM', '4'))


class TestMythWritingsNotifyFunctions:
    """Test the notify functions for myth_writings table."""
    
    @pytest.mark.integration
    def test_notify_myth_writings_insert_function_exists(self):
        """Test that the notify_myth_writings_insert function exists."""
        with session_scope() as session:
            result = session.execute(text("""
                SELECT routine_name, routine_type
                FROM information_schema.routines 
                WHERE routine_name = 'notify_myth_writings_insert'
                AND routine_schema = 'public'
            """))
            functions = result.fetchall()
            
            assert len(functions) == 1, "notify_myth_writings_insert function should exist"
            assert functions[0][1] == 'FUNCTION', "Should be a function"
    
    @pytest.mark.integration
    def test_notify_myth_writings_mark_read_function_exists(self):
        """Test that the notify_myth_writings_mark_read function exists."""
        with session_scope() as session:
            result = session.execute(text("""
                SELECT routine_name, routine_type
                FROM information_schema.routines 
                WHERE routine_name = 'notify_myth_writings_mark_read'
                AND routine_schema = 'public'
            """))
            functions = result.fetchall()
            
            assert len(functions) == 1, "notify_myth_writings_mark_read function should exist"
            assert functions[0][1] == 'FUNCTION', "Should be a function"
    
    @pytest.mark.integration
    def test_myth_writings_insert_trigger_exists(self):
        """Test that the myth_writings_insert_notify trigger exists."""
        with session_scope() as session:
            result = session.execute(text("""
                SELECT trigger_name, event_manipulation, action_timing
                FROM information_schema.triggers 
                WHERE trigger_name = 'myth_writings_insert_notify'
                AND trigger_schema = 'public'
            """))
            triggers = result.fetchall()
            
            assert len(triggers) == 1, "myth_writings_insert_notify trigger should exist"
            assert triggers[0][1] == 'INSERT', "Trigger should fire on INSERT"
            assert triggers[0][2] == 'AFTER', "Trigger should fire AFTER the event"
    
    @pytest.mark.integration
    def test_myth_writings_mark_read_trigger_exists(self):
        """Test that the myth_writings_mark_read_notify trigger exists."""
        with session_scope() as session:
            result = session.execute(text("""
                SELECT trigger_name, event_manipulation, action_timing
                FROM information_schema.triggers 
                WHERE trigger_name = 'myth_writings_mark_read_notify'
                AND trigger_schema = 'public'
            """))
            triggers = result.fetchall()
            
            assert len(triggers) == 1, "myth_writings_mark_read_notify trigger should exist"
            assert triggers[0][1] == 'UPDATE', "Trigger should fire on UPDATE"
            assert triggers[0][2] == 'AFTER', "Trigger should fire AFTER the event"
    
    @pytest.mark.integration
    def test_myth_writings_insert_creates_record(self):
        """Test that inserting a myth_writing creates a record with correct defaults."""
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
            
            # Insert a myth writing (this will trigger the notify function)
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            unique_text = f'Test myth writing {unique_id}'
            
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': unique_text
            })
            
            # Verify the record was created
            result = session.execute(text("""
                SELECT id, myth_id, written_text, has_been_read, created_at 
                FROM myth_writings 
                WHERE myth_id = :myth_id AND written_text = :written_text
            """), {
                'myth_id': myth_id,
                'written_text': unique_text
            })
            
            writing = result.fetchone()
            assert writing is not None, "Should have created a myth writing record"
            assert writing[1] == myth_id, "myth_id should match"
            assert writing[2] == unique_text, "written_text should match"
            assert writing[3] is False, "has_been_read should default to false"
            assert writing[4] is not None, "created_at should be set"
    
    @pytest.mark.integration
    def test_myth_writings_mark_read_updates_record(self):
        """Test that updating has_been_read to true works correctly."""
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
                'written_text': 'Test myth writing for mark read'
            })
            
            # Get the writing ID
            writing_result = session.execute(text("""
                SELECT id FROM myth_writings 
                WHERE myth_id = :myth_id
            """), {
                'myth_id': myth_id
            })
            writing_id = writing_result.fetchone()[0]
            
            # Update has_been_read to true (this will trigger the notify function)
            session.execute(text("""
                UPDATE myth_writings 
                SET has_been_read = true 
                WHERE id = :writing_id
            """), {
                'writing_id': writing_id
            })
            
            # Verify the update worked
            result = session.execute(text("""
                SELECT has_been_read FROM myth_writings 
                WHERE id = :writing_id
            """), {
                'writing_id': writing_id
            })
            
            writing = result.fetchone()
            assert writing is not None, "Should have found the myth writing"
            assert writing[0] is True, "has_been_read should be updated to true"
    
    @pytest.mark.integration
    def test_myth_writings_mark_read_no_change_when_already_true(self):
        """Test that updating has_been_read when already true doesn't cause issues."""
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
            
            # Insert a myth writing with has_been_read = true
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text, has_been_read) 
                VALUES (:myth_id, :written_text, :has_been_read)
            """), {
                'myth_id': myth_id,
                'written_text': 'Test myth writing already read',
                'has_been_read': True
            })
            
            # Get the writing ID
            writing_result = session.execute(text("""
                SELECT id FROM myth_writings 
                WHERE myth_id = :myth_id
            """), {
                'myth_id': myth_id
            })
            writing_id = writing_result.fetchone()[0]
            
            # Update has_been_read to true again (should not trigger notification)
            session.execute(text("""
                UPDATE myth_writings 
                SET has_been_read = true 
                WHERE id = :writing_id
            """), {
                'writing_id': writing_id
            })
            
            # Verify the record still exists and is unchanged
            result = session.execute(text("""
                SELECT has_been_read FROM myth_writings 
                WHERE id = :writing_id
            """), {
                'writing_id': writing_id
            })
            
            writing = result.fetchone()
            assert writing is not None, "Should have found the myth writing"
            assert writing[0] is True, "has_been_read should still be true"
    
    @pytest.mark.integration
    def test_myth_writings_has_been_read_column_exists(self):
        """Test that the has_been_read column exists in myth_writings table."""
        with session_scope() as session:
            result = session.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'myth_writings' 
                AND column_name = 'has_been_read'
            """))
            columns = result.fetchall()
            
            assert len(columns) == 1, "has_been_read column should exist"
            assert columns[0][1] == 'boolean', "has_been_read should be boolean type"
            assert columns[0][2] == 'NO', "has_been_read should be NOT NULL"
            assert columns[0][3] == 'false', "has_been_read should default to false"
    
    @pytest.mark.integration
    def test_myth_writings_has_been_read_default_value(self):
        """Test that myth_writings inserts default to has_been_read = false."""
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
            
            # Insert myth writing without specifying has_been_read
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            unique_text = f'Test myth writing default {unique_id}'
            
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': unique_text
            })
            
            # Check that has_been_read defaults to false
            result = session.execute(text("""
                SELECT has_been_read FROM myth_writings 
                WHERE myth_id = :myth_id AND written_text = :written_text
            """), {
                'myth_id': myth_id,
                'written_text': unique_text
            })
            
            writing = result.fetchone()
            assert writing is not None, "Should have inserted a myth writing"
            assert writing[0] is False, "has_been_read should default to false"
    
    @pytest.mark.integration
    def test_myth_writings_has_been_read_can_be_updated(self):
        """Test that has_been_read can be updated from false to true."""
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
            
            # Insert myth writing
            session.execute(text("""
                INSERT INTO myth_writings (myth_id, written_text) 
                VALUES (:myth_id, :written_text)
            """), {
                'myth_id': myth_id,
                'written_text': 'Test myth writing'
            })
            
            # Get the writing ID
            writing_result = session.execute(text("""
                SELECT id FROM myth_writings 
                WHERE myth_id = :myth_id
            """), {
                'myth_id': myth_id
            })
            writing_id = writing_result.fetchone()[0]
            
            # Update has_been_read to true
            session.execute(text("""
                UPDATE myth_writings 
                SET has_been_read = true 
                WHERE id = :writing_id
            """), {
                'writing_id': writing_id
            })
            
            # Verify the update
            result = session.execute(text("""
                SELECT has_been_read FROM myth_writings 
                WHERE id = :writing_id
            """), {
                'writing_id': writing_id
            })
            
            writing = result.fetchone()
            assert writing is not None, "Should have found the myth writing"
            assert writing[0] is True, "has_been_read should be updated to true"
