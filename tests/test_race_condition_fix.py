"""
Test for race condition fix in agent_myths table.
"""

import random
import threading
import pytest
from mythologizer_postgres.connectors import insert_agent_myth_safe
from mythologizer_postgres.db import psycopg_connection, clear_all_rows


def create_test_data():
    """Create test agents and myths."""
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # Create test agent
            cur.execute("""
                INSERT INTO agents (name, memory_size) 
                VALUES ('Test Agent', 5)
                RETURNING id
            """)
            agent_id = cur.fetchone()[0]
            
            # Create test myths
            myth_ids = []
            for i in range(10):
                cur.execute("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (%s, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
                    RETURNING id
                """, ([random.random() for _ in range(4)],))
                myth_ids.append(cur.fetchone()[0])
            
            conn.commit()
            return agent_id, myth_ids


def concurrent_insert_worker(agent_id, myth_ids, results, worker_id):
    """Worker function to perform concurrent inserts."""
    for myth_id in myth_ids:
        try:
            success = insert_agent_myth_safe(
                myth_id=myth_id,
                agent_id=agent_id,
                retention=random.uniform(0.1, 1.0)
            )
            results.append((worker_id, myth_id, success))
        except Exception as e:
            results.append((worker_id, myth_id, False, str(e)))


@pytest.mark.integration
def test_concurrent_inserts_race_condition_fix():
    """Test concurrent inserts to verify race condition fix."""
    # Clear existing data (tables already exist from test setup)
    clear_all_rows()
    
    # Create test data
    agent_id, myth_ids = create_test_data()
    
    # Test concurrent inserts
    results = []
    threads = []
    
    # Create multiple threads to simulate concurrent access
    for i in range(3):
        thread_myth_ids = myth_ids[i*3:(i+1)*3]  # Distribute myths among threads
        thread = threading.Thread(
            target=concurrent_insert_worker,
            args=(agent_id, thread_myth_ids, results, i)
        )
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Analyze results
    successful_inserts = [r for r in results if r[2]]
    failed_inserts = [r for r in results if not r[2]]
    
    # Verify final state
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM agent_myths WHERE agent_id = %s
            """, (agent_id,))
            final_count = cur.fetchone()[0]
            
            cur.execute("""
                SELECT position FROM agent_myths 
                WHERE agent_id = %s 
                ORDER BY position
            """, (agent_id,))
            positions = [row[0] for row in cur.fetchall()]
    
    # Assertions
    assert len(successful_inserts) > 0, "Should have some successful inserts"
    assert len(positions) == len(set(positions)), "No duplicate positions should exist"
    
    # Check that positions are contiguous starting from 0
    expected_positions = list(range(len(positions)))
    assert positions == expected_positions, f"Positions should be contiguous starting from 0, got {positions}"


@pytest.mark.integration
def test_duplicate_myth_handling():
    """Test that inserting the same myth multiple times is handled correctly."""
    # Clear existing data (tables already exist from test setup)
    clear_all_rows()
    
    # Create test data
    agent_id, myth_ids = create_test_data()
    
    # Test inserting the same myth multiple times (should fail gracefully)
    myth_id = myth_ids[0]
    
    # First insert should succeed
    success1 = insert_agent_myth_safe(
        myth_id=myth_id,
        agent_id=agent_id,
        retention=0.8
    )
    assert success1, "First insert should succeed"
    
    # Second insert of same myth should fail gracefully
    success2 = insert_agent_myth_safe(
        myth_id=myth_id,
        agent_id=agent_id,
        retention=0.9
    )
    assert not success2, "Second insert of same myth should fail gracefully"


@pytest.mark.integration
def test_memory_size_limit():
    """Test that memory size limits are respected."""
    # Clear existing data (tables already exist from test setup)
    clear_all_rows()
    
    # Create test data with small memory size
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agents (name, memory_size) 
                VALUES ('Small Memory Agent', 3)
                RETURNING id
            """)
            agent_id = cur.fetchone()[0]
            
            # Create 5 myths
            myth_ids = []
            for i in range(5):
                cur.execute("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (%s, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
                    RETURNING id
                """, ([random.random() for _ in range(4)],))
                myth_ids.append(cur.fetchone()[0])
            
            conn.commit()
    
    # Insert all 5 myths (should only keep 3 due to memory size limit)
    for myth_id in myth_ids:
        success = insert_agent_myth_safe(
            myth_id=myth_id,
            agent_id=agent_id,
            retention=random.uniform(0.1, 1.0)
        )
        assert success, f"Insert of myth {myth_id} should succeed"
    
    # Verify only 3 myths remain
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM agent_myths WHERE agent_id = %s
            """, (agent_id,))
            final_count = cur.fetchone()[0]
    
    assert final_count == 3, f"Should only have 3 myths in memory, got {final_count}"

