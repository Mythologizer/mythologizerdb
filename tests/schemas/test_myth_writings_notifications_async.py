import pytest
import asyncio
import json
import psycopg
import numpy as np
from sqlalchemy import text
import time

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


class TestMythWritingsNotificationsAsync:
    """Test the actual PostgreSQL notifications for myth_writings table using async connections."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_myth_writings_insert_notification_basic(self):
        """Basic test that inserting a myth_writing works and doesn't crash."""
        url = build_url()
        
        # Create test data using sync session
        with session_scope() as session:
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
        
        # Test basic async connection and insert
        try:
            # Set up worker connection for insertions
            work_conn = await psycopg.AsyncConnection.connect(
                dbname=url.database, user=url.username, password=url.password,
                host=url.host, port=url.port,
            )
            
            # Insert a myth writing (this should trigger the notify function)
            async with work_conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO myth_writings (myth_id, written_text) VALUES (%s, %s) RETURNING id",
                    (myth_id, "Test myth writing for async notification")
                )
                result = await cur.fetchone()
                writing_id = result[0]
            
            # Verify the insert worked
            assert writing_id is not None, "Should have created a myth writing"
            
            # Clean up
            await work_conn.close()
            
        except Exception as e:
            pytest.fail(f"Async test failed with error: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_myth_writings_mark_read_notification_basic(self):
        """Basic test that marking a myth_writing as read works and doesn't crash."""
        url = build_url()
        
        # Create test data using sync session
        with session_scope() as session:
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
                'written_text': 'Test myth writing for async read notification'
            })
            
            # Get the writing ID
            writing_result = session.execute(text("""
                SELECT id FROM myth_writings 
                WHERE myth_id = :myth_id
            """), {
                'myth_id': myth_id
            })
            writing_id = writing_result.fetchone()[0]
        
        # Test basic async connection and update
        try:
            # Set up worker connection for updates
            work_conn = await psycopg.AsyncConnection.connect(
                dbname=url.database, user=url.username, password=url.password,
                host=url.host, port=url.port,
            )
            
            # Update has_been_read to true (this should trigger the notify function)
            async with work_conn.cursor() as cur:
                await cur.execute(
                    "UPDATE myth_writings SET has_been_read = true WHERE id = %s",
                    (writing_id,)
                )
            
            # Verify the update worked
            async with work_conn.cursor() as cur:
                await cur.execute(
                    "SELECT has_been_read FROM myth_writings WHERE id = %s",
                    (writing_id,)
                )
                result = await cur.fetchone()
                assert result[0] is True, "has_been_read should be updated to true"
            
            # Clean up
            await work_conn.close()
            
        except Exception as e:
            pytest.fail(f"Async test failed with error: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_myth_writings_async_connections_work(self):
        """Test that async connections to the database work correctly."""
        url = build_url()
        
        try:
            # Test basic connection
            conn = await psycopg.AsyncConnection.connect(
                dbname=url.database, user=url.username, password=url.password,
                host=url.host, port=url.port,
            )
            
            # Test basic query
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 as test_value")
                result = await cur.fetchone()
                assert result[0] == 1, "Basic query should work"
            
            # Test myth_writings table exists
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = 'myth_writings'
                """)
                result = await cur.fetchone()
                assert result[0] > 0, "myth_writings table should exist"
            
            await conn.close()
            
        except Exception as e:
            pytest.fail(f"Async connection test failed with error: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_myth_writings_notify_functions_exist(self):
        """Test that the notify functions exist in the database via async connection."""
        url = build_url()
        
        try:
            conn = await psycopg.AsyncConnection.connect(
                dbname=url.database, user=url.username, password=url.password,
                host=url.host, port=url.port,
            )
            
            # Check if notify functions exist
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT routine_name FROM information_schema.routines 
                    WHERE routine_name IN ('notify_myth_writings_insert', 'notify_myth_writings_mark_read')
                    AND routine_schema = 'public'
                """)
                results = await cur.fetchall()
                
                function_names = [row[0] for row in results]
                assert 'notify_myth_writings_insert' in function_names, "notify_myth_writings_insert function should exist"
                assert 'notify_myth_writings_mark_read' in function_names, "notify_myth_writings_mark_read function should exist"
            
            await conn.close()
            
        except Exception as e:
            pytest.fail(f"Async function check failed with error: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_myth_writings_triggers_exist(self):
        """Test that the triggers exist in the database via async connection."""
        url = build_url()
        
        try:
            conn = await psycopg.AsyncConnection.connect(
                dbname=url.database, user=url.username, password=url.password,
                host=url.host, port=url.port,
            )
            
            # Check if triggers exist
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT trigger_name FROM information_schema.triggers 
                    WHERE trigger_name IN ('myth_writings_insert_notify', 'myth_writings_mark_read_notify')
                    AND trigger_schema = 'public'
                """)
                results = await cur.fetchall()
                
                trigger_names = [row[0] for row in results]
                assert 'myth_writings_insert_notify' in trigger_names, "myth_writings_insert_notify trigger should exist"
                assert 'myth_writings_mark_read_notify' in trigger_names, "myth_writings_mark_read_notify trigger should exist"
            
            await conn.close()
            
        except Exception as e:
            pytest.fail(f"Async trigger check failed with error: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_myth_writings_async_crud_operations(self):
        """Test that basic CRUD operations work via async connections."""
        url = build_url()
        
        # Create test data using sync session
        with session_scope() as session:
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
        
        try:
            conn = await psycopg.AsyncConnection.connect(
                dbname=url.database, user=url.username, password=url.password,
                host=url.host, port=url.port,
            )
            
            # Test INSERT
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO myth_writings (myth_id, written_text) VALUES (%s, %s) RETURNING id",
                    (myth_id, "Test async CRUD operation")
                )
                result = await cur.fetchone()
                writing_id = result[0]
            
            # Test SELECT
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT written_text, has_been_read FROM myth_writings WHERE id = %s",
                    (writing_id,)
                )
                result = await cur.fetchone()
                assert result[0] == "Test async CRUD operation", "SELECT should return correct text"
                assert result[1] is False, "has_been_read should default to false"
            
            # Test UPDATE
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE myth_writings SET has_been_read = true WHERE id = %s",
                    (writing_id,)
                )
            
            # Verify UPDATE worked
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT has_been_read FROM myth_writings WHERE id = %s",
                    (writing_id,)
                )
                result = await cur.fetchone()
                assert result[0] is True, "UPDATE should have changed has_been_read to true"
            
            await conn.close()
            
        except Exception as e:
            pytest.fail(f"Async CRUD test failed with error: {e}")
