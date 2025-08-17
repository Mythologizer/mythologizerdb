#!/usr/bin/env python3
"""
Simple test runner for myth_writings notify functions.
This script can be run independently to test the notify functionality.
"""

import asyncio
import json
import psycopg
import numpy as np
from sqlalchemy import text
import os
import sys

# Add the parent directory to the path so we can import mythologizer_postgres
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mythologizer_postgres.db import (
    session_scope,
    build_url,
    get_table_row_counts,
    clear_all_rows,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


async def test_notify_functions():
    """Test the notify functions manually."""
    print("Testing myth_writings notify functions...")
    
    url = build_url()
    print(f"Connecting to database: {url.host}:{url.port}/{url.database}")
    
    # Create test data using sync session
    with session_scope() as session:
        print("Creating test data...")
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
        print(f"Created myth with ID: {myth_id}")
    
    # Test 1: Insert notification
    print("\n=== Test 1: Insert Notification ===")
    await test_insert_notification(url, myth_id)
    
    # Test 2: Mark as read notification
    print("\n=== Test 2: Mark as Read Notification ===")
    await test_mark_read_notification(url, myth_id)
    
    # Test 3: No notification when already read
    print("\n=== Test 3: No Notification When Already Read ===")
    await test_no_notification_when_already_read(url, myth_id)
    
    print("\nAll tests completed!")


async def test_insert_notification(url, myth_id):
    """Test that inserting a myth_writing triggers a notification."""
    # Set up async listener connection
    listen_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    await listen_conn.set_autocommit(True)
    await listen_conn.execute("LISTEN myth_writings_inserted")
    print("Listening on myth_writings_inserted")
    
    # Set up worker connection for insertions
    work_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    
    notifications_received = []
    
    # Start listening for notifications
    async def listen_for_notifications():
        async for notify in listen_conn.notifies():
            notifications_received.append(notify)
            print(f"Received notification: {notify.channel} - {notify.payload}")
            if len(notifications_received) >= 1:
                break
    
    # Start the listener task
    listener_task = asyncio.create_task(listen_for_notifications())
    
    # Wait a bit for the listener to be ready
    await asyncio.sleep(0.1)
    
    # Insert a myth writing (this should trigger the notification)
    async with work_conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO myth_writings (myth_id, written_text) VALUES (%s, %s) RETURNING id",
            (myth_id, "Test myth writing for notification")
        )
        result = await cur.fetchone()
        writing_id = result[0]
        print(f"Inserted myth writing with ID: {writing_id}")
    
    # Wait for notification with timeout
    try:
        await asyncio.wait_for(listener_task, timeout=5.0)
        print("✓ Insert notification received successfully")
    except asyncio.TimeoutError:
        print("✗ Timeout waiting for insert notification")
    
    # Clean up
    await listen_conn.close()
    await work_conn.close()


async def test_mark_read_notification(url, myth_id):
    """Test that marking a myth_writing as read triggers a notification."""
    # Create a myth writing first
    with session_scope() as session:
        session.execute(text("""
            INSERT INTO myth_writings (myth_id, written_text) 
            VALUES (:myth_id, :written_text)
        """), {
            'myth_id': myth_id,
            'written_text': 'Test myth writing for read notification'
        })
        
        # Get the writing ID
        writing_result = session.execute(text("""
            SELECT id FROM myth_writings 
            WHERE myth_id = :myth_id AND written_text = :written_text
        """), {
            'myth_id': myth_id,
            'written_text': 'Test myth writing for read notification'
        })
        writing_id = writing_result.fetchone()[0]
        print(f"Created myth writing with ID: {writing_id}")
    
    # Set up async listener connection
    listen_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    await listen_conn.set_autocommit(True)
    await listen_conn.execute("LISTEN myth_writings_marked_read")
    print("Listening on myth_writings_marked_read")
    
    # Set up worker connection for updates
    work_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    
    notifications_received = []
    
    # Start listening for notifications
    async def listen_for_notifications():
        async for notify in listen_conn.notifies():
            notifications_received.append(notify)
            print(f"Received notification: {notify.channel} - {notify.payload}")
            if len(notifications_received) >= 1:
                break
    
    # Start the listener task
    listener_task = asyncio.create_task(listen_for_notifications())
    
    # Wait a bit for the listener to be ready
    await asyncio.sleep(0.1)
    
    # Update has_been_read to true (this should trigger the notification)
    async with work_conn.cursor() as cur:
        await cur.execute(
            "UPDATE myth_writings SET has_been_read = true WHERE id = %s",
            (writing_id,)
        )
        print(f"Updated myth writing {writing_id} to has_been_read = true")
    
    # Wait for notification with timeout
    try:
        await asyncio.wait_for(listener_task, timeout=5.0)
        print("✓ Mark as read notification received successfully")
    except asyncio.TimeoutError:
        print("✗ Timeout waiting for mark as read notification")
    
    # Clean up
    await listen_conn.close()
    await work_conn.close()


async def test_no_notification_when_already_read(url, myth_id):
    """Test that updating has_been_read when already true doesn't trigger notification."""
    # Create a myth writing with has_been_read = true
    with session_scope() as session:
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
            WHERE myth_id = :myth_id AND written_text = :written_text
        """), {
            'myth_id': myth_id,
            'written_text': 'Test myth writing already read'
        })
        writing_id = writing_result.fetchone()[0]
        print(f"Created myth writing with ID: {writing_id} (already read)")
    
    # Set up async listener connection
    listen_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    await listen_conn.set_autocommit(True)
    await listen_conn.execute("LISTEN myth_writings_marked_read")
    print("Listening on myth_writings_marked_read")
    
    # Set up worker connection for updates
    work_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    
    notifications_received = []
    
    # Start listening for notifications
    async def listen_for_notifications():
        async for notify in listen_conn.notifies():
            notifications_received.append(notify)
            print(f"Received notification: {notify.channel} - {notify.payload}")
    
    # Start the listener task
    listener_task = asyncio.create_task(listen_for_notifications())
    
    # Wait a bit for the listener to be ready
    await asyncio.sleep(0.1)
    
    # Update has_been_read to true again (this should NOT trigger notification)
    async with work_conn.cursor() as cur:
        await cur.execute(
            "UPDATE myth_writings SET has_been_read = true WHERE id = %s",
            (writing_id,)
        )
        print(f"Updated myth writing {writing_id} to has_been_read = true again")
    
    # Wait a bit to see if any notifications come in
    await asyncio.sleep(2.0)
    
    # Cancel the listener task
    listener_task.cancel()
    
    # Verify we received NO notifications
    if len(notifications_received) == 0:
        print("✓ No notification received when updating already-read writing (correct behavior)")
    else:
        print(f"✗ Unexpected notifications received: {len(notifications_received)}")
    
    # Clean up
    await listen_conn.close()
    await work_conn.close()


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_notify_functions())
