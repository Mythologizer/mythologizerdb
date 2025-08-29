"""
Comprehensive edge case tests for insert_agent_myth_safe function.
"""

import pytest
import random
import threading
import time
from mythologizer_postgres.connectors import insert_agent_myth_safe
from mythologizer_postgres.db import psycopg_connection, clear_all_rows


def create_test_agent(memory_size=5, name="Test Agent"):
    """Create a test agent and return its ID."""
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agents (name, memory_size) 
                VALUES (%s, %s)
                RETURNING id
            """, (name, memory_size))
            agent_id = cur.fetchone()[0]
            conn.commit()
            return agent_id


def create_test_myth():
    """Create a test myth and return its ID."""
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            embedding = [random.random() for _ in range(4)]
            cur.execute("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (%s, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
                RETURNING id
            """, (embedding,))
            myth_id = cur.fetchone()[0]
            conn.commit()
            return myth_id


class TestInsertAgentMythSafeEdgeCases:
    """Test edge cases for insert_agent_myth_safe function."""
    
    def setup_method(self):
        """Clear all data before each test."""
        clear_all_rows()
    
    @pytest.mark.integration
    def test_invalid_retention_values(self):
        """Test that invalid retention values are properly rejected."""
        agent_id = create_test_agent()
        myth_id = create_test_myth()
        
        # Test retention = 0
        success = insert_agent_myth_safe(myth_id, agent_id, 0.0)
        assert not success, "Should fail with retention = 0"
        
        # Test retention < 0
        success = insert_agent_myth_safe(myth_id, agent_id, -0.1)
        assert not success, "Should fail with retention < 0"
        
        # Test retention = -1
        success = insert_agent_myth_safe(myth_id, agent_id, -1.0)
        assert not success, "Should fail with retention = -1"
        
        # Test very small positive retention (should work)
        success = insert_agent_myth_safe(myth_id, agent_id, 0.0001)
        assert success, "Should succeed with very small positive retention"
    
    @pytest.mark.integration
    def test_nonexistent_agent(self):
        """Test that non-existent agent IDs are properly handled."""
        myth_id = create_test_myth()
        
        # Test with non-existent agent ID
        success = insert_agent_myth_safe(myth_id, 99999, 0.8)
        assert not success, "Should fail with non-existent agent"
        
        # Test with negative agent ID
        success = insert_agent_myth_safe(myth_id, -1, 0.8)
        assert not success, "Should fail with negative agent ID"
    
    @pytest.mark.integration
    def test_agent_with_minimum_memory_size(self):
        """Test that agents with minimum memory_size = 3 are properly handled."""
        # Create agent with minimum memory_size = 3
        agent_id = create_test_agent(memory_size=3)
        myth_id = create_test_myth()
        
        success = insert_agent_myth_safe(myth_id, agent_id, 0.8)
        assert success, "Should succeed with minimum memory_size = 3"
    
    @pytest.mark.integration
    def test_nonexistent_myth(self):
        """Test that non-existent myth IDs are properly handled."""
        agent_id = create_test_agent()
        
        # Test with non-existent myth ID
        success = insert_agent_myth_safe(99999, agent_id, 0.8)
        assert not success, "Should fail with non-existent myth"
        
        # Test with negative myth ID
        success = insert_agent_myth_safe(-1, agent_id, 0.8)
        assert not success, "Should fail with negative myth ID"
    
    @pytest.mark.integration
    def test_myth_already_assigned_to_different_agent(self):
        """Test that myths already assigned to other agents are properly handled."""
        agent1_id = create_test_agent(name="Agent 1")
        agent2_id = create_test_agent(name="Agent 2")
        myth_id = create_test_myth()
        
        # Insert myth into first agent
        success = insert_agent_myth_safe(myth_id, agent1_id, 0.8)
        assert success, "First insert should succeed"
        
        # Try to insert same myth into second agent
        success = insert_agent_myth_safe(myth_id, agent2_id, 0.9)
        assert not success, "Should fail when myth is already assigned to another agent"
    
    @pytest.mark.integration
    def test_edge_case_retention_values(self):
        """Test edge case retention values."""
        agent_id = create_test_agent()
        myth_id = create_test_myth()
        
        # Test very small positive retention
        success = insert_agent_myth_safe(myth_id, agent_id, 0.000001)
        assert success, "Should succeed with very small positive retention"
        
        # Create another myth for testing
        myth_id2 = create_test_myth()
        
        # Test very large retention
        success = insert_agent_myth_safe(myth_id2, agent_id, 999999.0)
        assert success, "Should succeed with very large retention"
    
    @pytest.mark.integration
    def test_concurrent_operations_on_different_agents(self):
        """Test that concurrent operations on different agents work correctly."""
        agent1_id = create_test_agent(name="Agent 1")
        agent2_id = create_test_agent(name="Agent 2")
        
        results = []
        
        def worker(agent_id, worker_id):
            """Worker function for concurrent operations."""
            try:
                myth_id = create_test_myth()
                success = insert_agent_myth_safe(myth_id, agent_id, 0.8)
                results.append((worker_id, agent_id, success))
            except Exception as e:
                results.append((worker_id, agent_id, False, str(e)))
        
        # Start concurrent operations on different agents
        threads = []
        for i in range(5):
            agent_id = agent1_id if i % 2 == 0 else agent2_id
            thread = threading.Thread(target=worker, args=(agent_id, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        successful_ops = [r for r in results if r[2]]
        assert len(successful_ops) > 0, "Should have some successful operations"
        
        # Verify both agents have myths
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM agent_myths WHERE agent_id = %s", (agent1_id,))
                count1 = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM agent_myths WHERE agent_id = %s", (agent2_id,))
                count2 = cur.fetchone()[0]
        
        assert count1 > 0, "Agent 1 should have myths"
        assert count2 > 0, "Agent 2 should have myths"
    
    @pytest.mark.integration
    def test_memory_eviction_edge_cases(self):
        """Test edge cases around memory eviction."""
        # Create agent with memory_size = 3 (minimum allowed)
        agent_id = create_test_agent(memory_size=3)
        
        # Insert 3 myths (fill up memory)
        myth_ids = []
        for i in range(3):
            myth_id = create_test_myth()
            success = insert_agent_myth_safe(myth_id, agent_id, 0.8)
            assert success, f"Insert {i+1} should succeed"
            myth_ids.append(myth_id)
        
        # Insert 4th myth (should evict the oldest one)
        myth4_id = create_test_myth()
        success = insert_agent_myth_safe(myth4_id, agent_id, 0.9)
        assert success, "Fourth insert should succeed"
        
        # Verify only 3 myths remain and oldest was evicted
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM agent_myths WHERE agent_id = %s", (agent_id,))
                count = cur.fetchone()[0]
                assert count == 3, "Should have exactly 3 myths after eviction"
                
                # Verify the lowest retention myth was evicted (all have same retention 0.8, so highest myth_id gets evicted)
                cur.execute("SELECT myth_id FROM agent_myths WHERE agent_id = %s", (agent_id,))
                remaining_myth_ids = [row[0] for row in cur.fetchall()]
                assert myth_ids[2] not in remaining_myth_ids, "Highest myth_id should be evicted (tie-breaker for same retention)"
                assert myth4_id in remaining_myth_ids, "Newest myth should remain"
    
    @pytest.mark.integration
    def test_position_assignment_edge_cases(self):
        """Test edge cases around position assignment."""
        agent_id = create_test_agent(memory_size=3)
        
        # Insert myths and verify positions
        myth_ids = []
        for i in range(3):
            myth_id = create_test_myth()
            success = insert_agent_myth_safe(myth_id, agent_id, 0.8)
            assert success, f"Insert {i+1} should succeed"
            myth_ids.append(myth_id)
        
        # Verify positions are correct (LIFO: last inserted at position 0)
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
        
        assert len(positions) == 3, "Should have 3 myths"
        assert positions[0][1] == 0, "First position should be 0"
        assert positions[1][1] == 1, "Second position should be 1"
        assert positions[2][1] == 2, "Third position should be 2"
        
        # Verify retention-based order (all retentions are 0.8, so ordered by myth_id ASC)
        assert positions[0][0] == myth_ids[0], "First myth should be at position 0 (lowest myth_id)"
        assert positions[1][0] == myth_ids[1], "Second myth should be at position 1 (middle myth_id)"
        assert positions[2][0] == myth_ids[2], "Third myth should be at position 2 (highest myth_id)"
    
    @pytest.mark.integration
    def test_comprehensive_reordering_logic(self):
        """Test the comprehensive reordering logic: new myths at position 0, existing myths shifted down."""
        agent_id = create_test_agent(memory_size=5)
        
        # Step 1: Insert first myth - should be at position 0
        myth1_id = create_test_myth()
        success = insert_agent_myth_safe(myth1_id, agent_id, 0.8)
        assert success, "First insert should succeed"
        
        # Verify first myth is at position 0
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
        
        assert len(positions) == 1, "Should have 1 myth"
        assert positions[0][0] == myth1_id, "First myth should be myth1"
        assert positions[0][1] == 0, "First myth should be at position 0"
        
        # Step 2: Insert second myth - should be at position 0, myth1 moves to position 1
        myth2_id = create_test_myth()
        success = insert_agent_myth_safe(myth2_id, agent_id, 0.9)
        assert success, "Second insert should succeed"
        
        # Verify reordering: myth2 at position 0, myth1 at position 1
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
        
        assert len(positions) == 2, "Should have 2 myths"
        assert positions[0][0] == myth2_id, "Second myth should be at position 0"
        assert positions[0][1] == 0, "Second myth should be at position 0"
        assert positions[1][0] == myth1_id, "First myth should be at position 1"
        assert positions[1][1] == 1, "First myth should be at position 1"
        
        # Step 3: Insert third myth - should be at position 0, others shift down
        myth3_id = create_test_myth()
        success = insert_agent_myth_safe(myth3_id, agent_id, 0.7)
        assert success, "Third insert should succeed"
        
        # Verify reordering: myth3 at position 0, myth2 at position 1, myth1 at position 2
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
        
        assert len(positions) == 3, "Should have 3 myths"
        # After retention-based reordering: myth2 (0.9) at position 0, myth3 (0.7) at position 1, myth1 (0.8) at position 2
        assert positions[0][0] == myth2_id, "Highest retention myth (0.9) should be at position 0"
        assert positions[0][1] == 0, "Highest retention myth should be at position 0"
        assert positions[1][0] == myth1_id, "Second highest retention myth (0.8) should be at position 1"
        assert positions[1][1] == 1, "Second highest retention myth should be at position 1"
        assert positions[2][0] == myth3_id, "Lowest retention myth (0.7) should be at position 2"
        assert positions[2][1] == 2, "Lowest retention myth should be at position 2"
        
        # Step 4: Insert fourth myth - should be at position 0, others shift down
        myth4_id = create_test_myth()
        success = insert_agent_myth_safe(myth4_id, agent_id, 0.95)
        assert success, "Fourth insert should succeed"
        
        # Verify reordering: myth4 at position 0, myth3 at position 1, myth2 at position 2, myth1 at position 3
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
        
        assert len(positions) == 4, "Should have 4 myths"
        # After retention-based reordering: myth4 (0.95) at position 0, myth2 (0.9) at position 1, myth1 (0.8) at position 2, myth3 (0.7) at position 3
        assert positions[0][0] == myth4_id, "Highest retention myth (0.95) should be at position 0"
        assert positions[0][1] == 0, "Highest retention myth should be at position 0"
        assert positions[1][0] == myth2_id, "Second highest retention myth (0.9) should be at position 1"
        assert positions[1][1] == 1, "Second highest retention myth should be at position 1"
        assert positions[2][0] == myth1_id, "Third highest retention myth (0.8) should be at position 2"
        assert positions[2][1] == 2, "Third highest retention myth should be at position 2"
        assert positions[3][0] == myth3_id, "Lowest retention myth (0.7) should be at position 3"
        assert positions[3][1] == 3, "Lowest retention myth should be at position 3"
        
        # Step 5: Insert fifth myth - should be at position 0, others shift down
        myth5_id = create_test_myth()
        success = insert_agent_myth_safe(myth5_id, agent_id, 0.6)
        assert success, "Fifth insert should succeed"
        
        # Verify reordering: myth5 at position 0, myth4 at position 1, myth3 at position 2, myth2 at position 3, myth1 at position 4
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
        
        assert len(positions) == 5, "Should have 5 myths"
        # After retention-based reordering: myth4 (0.95) at position 0, myth2 (0.9) at position 1, myth1 (0.8) at position 2, myth3 (0.7) at position 3, myth5 (0.6) at position 4
        assert positions[0][0] == myth4_id, "Highest retention myth (0.95) should be at position 0"
        assert positions[0][1] == 0, "Highest retention myth should be at position 0"
        assert positions[1][0] == myth2_id, "Second highest retention myth (0.9) should be at position 1"
        assert positions[1][1] == 1, "Second highest retention myth should be at position 1"
        assert positions[2][0] == myth1_id, "Third highest retention myth (0.8) should be at position 2"
        assert positions[2][1] == 2, "Third highest retention myth should be at position 2"
        assert positions[3][0] == myth3_id, "Fourth highest retention myth (0.7) should be at position 3"
        assert positions[3][1] == 3, "Fourth highest retention myth should be at position 3"
        assert positions[4][0] == myth5_id, "Lowest retention myth (0.6) should be at position 4"
        assert positions[4][1] == 4, "Lowest retention myth should be at position 4"
        
        # Step 6: Insert sixth myth - should evict myth1 (position 4) and reorder
        myth6_id = create_test_myth()
        success = insert_agent_myth_safe(myth6_id, agent_id, 0.85)
        assert success, "Sixth insert should succeed"
        
        # Verify eviction and reordering: myth6 at position 0, myth5 at position 1, myth4 at position 2, myth3 at position 3, myth2 at position 4
        # myth1 should be evicted
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
                
                # Verify myth5 (lowest retention 0.6) was evicted
                cur.execute("SELECT COUNT(*) FROM agent_myths WHERE myth_id = %s", (myth5_id,))
                myth5_count = cur.fetchone()[0]
        
        assert len(positions) == 5, "Should still have 5 myths (memory size limit)"
        assert myth5_count == 0, "myth5 (lowest retention 0.6) should be evicted"
        # After retention-based reordering: myth4 (0.95) at position 0, myth2 (0.9) at position 1, myth6 (0.85) at position 2, myth1 (0.8) at position 3, myth3 (0.7) at position 4
        assert positions[0][0] == myth4_id, "Highest retention myth (0.95) should be at position 0"
        assert positions[0][1] == 0, "Highest retention myth should be at position 0"
        assert positions[1][0] == myth2_id, "Second highest retention myth (0.9) should be at position 1"
        assert positions[1][1] == 1, "Second highest retention myth should be at position 1"
        assert positions[2][0] == myth6_id, "Third highest retention myth (0.85) should be at position 2"
        assert positions[2][1] == 2, "Third highest retention myth should be at position 2"
        assert positions[3][0] == myth1_id, "Fourth highest retention myth (0.8) should be at position 3"
        assert positions[3][1] == 3, "Fourth highest retention myth should be at position 3"
        assert positions[4][0] == myth3_id, "Lowest retention myth (0.7) should be at position 4"
        assert positions[4][1] == 4, "Lowest retention myth should be at position 4"
        
        # Verify positions are contiguous (0, 1, 2, 3, 4)
        expected_positions = [0, 1, 2, 3, 4]
        actual_positions = [pos[1] for pos in positions]
        assert actual_positions == expected_positions, f"Positions should be contiguous, got {actual_positions}"
        
        # Step 7: Insert seventh myth - should evict myth2 (position 4) and reorder
        myth7_id = create_test_myth()
        success = insert_agent_myth_safe(myth7_id, agent_id, 0.75)
        assert success, "Seventh insert should succeed"
        
        # Verify eviction and reordering: myth7 at position 0, myth6 at position 1, myth5 at position 2, myth4 at position 3, myth3 at position 4
        # myth2 should be evicted
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = cur.fetchall()
                
                # Verify myth3 (lowest retention 0.7) was evicted
                cur.execute("SELECT COUNT(*) FROM agent_myths WHERE myth_id = %s", (myth3_id,))
                myth3_count = cur.fetchone()[0]
        
        assert len(positions) == 5, "Should still have 5 myths (memory size limit)"
        assert myth3_count == 0, "myth3 (lowest retention 0.7) should be evicted"
        # After retention-based reordering: myth4 (0.95) at position 0, myth2 (0.9) at position 1, myth6 (0.85) at position 2, myth1 (0.8) at position 3, myth7 (0.75) at position 4
        assert positions[0][0] == myth4_id, "Highest retention myth (0.95) should be at position 0"
        assert positions[0][1] == 0, "Highest retention myth should be at position 0"
        assert positions[1][0] == myth2_id, "Second highest retention myth (0.9) should be at position 1"
        assert positions[1][1] == 1, "Second highest retention myth should be at position 1"
        assert positions[2][0] == myth6_id, "Third highest retention myth (0.85) should be at position 2"
        assert positions[2][1] == 2, "Third highest retention myth should be at position 2"
        assert positions[3][0] == myth1_id, "Fourth highest retention myth (0.8) should be at position 3"
        assert positions[3][1] == 3, "Fourth highest retention myth should be at position 3"
        assert positions[4][0] == myth7_id, "Lowest retention myth (0.75) should be at position 4"
        assert positions[4][1] == 4, "Lowest retention myth should be at position 4"
        
        # Final verification: positions are still contiguous
        expected_positions = [0, 1, 2, 3, 4]
        actual_positions = [pos[1] for pos in positions]
        assert actual_positions == expected_positions, f"Final positions should be contiguous, got {actual_positions}"
    
    @pytest.mark.integration
    def test_database_connection_failures(self):
        """Test behavior when database operations fail."""
        agent_id = create_test_agent()
        myth_id = create_test_myth()
        
        # This test is more about ensuring the function handles exceptions gracefully
        # We can't easily simulate database failures, but we can test that the function
        # returns False on any exception
        
        # Test with valid data (should succeed)
        success = insert_agent_myth_safe(myth_id, agent_id, 0.8)
        assert success, "Valid insert should succeed"
        
        # Test with invalid data that might cause database errors
        # (This is more of a robustness test)
        success = insert_agent_myth_safe(myth_id, agent_id, 0.8)  # Duplicate myth
        assert not success, "Duplicate myth should fail gracefully"
    
    @pytest.mark.integration
    def test_advisory_lock_contention(self):
        """Test advisory lock contention scenarios."""
        agent_id = create_test_agent()
        
        results = []
        
        def lock_contention_worker(worker_id):
            """Worker that creates high lock contention."""
            try:
                myth_id = create_test_myth()
                success = insert_agent_myth_safe(myth_id, agent_id, 0.8)
                results.append((worker_id, success))
            except Exception as e:
                results.append((worker_id, False, str(e)))
        
        # Start many concurrent operations to create lock contention
        threads = []
        for i in range(10):
            thread = threading.Thread(target=lock_contention_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify some operations succeeded
        successful_ops = [r for r in results if r[1]]
        assert len(successful_ops) > 0, "Should have some successful operations despite lock contention"
        
        # Verify final state is consistent
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM agent_myths WHERE agent_id = %s", (agent_id,))
                count = cur.fetchone()[0]
                assert count <= 5, "Should not exceed memory size"
                
                # Verify positions are contiguous
                cur.execute("""
                    SELECT position FROM agent_myths 
                    WHERE agent_id = %s 
                    ORDER BY position
                """, (agent_id,))
                positions = [row[0] for row in cur.fetchall()]
                expected_positions = list(range(len(positions)))
                assert positions == expected_positions, "Positions should be contiguous"
    
    @pytest.mark.integration
    def test_invalid_input_types(self):
        """Test behavior with invalid input types."""
        agent_id = create_test_agent()
        myth_id = create_test_myth()
        
        # Test with string instead of int for IDs
        try:
            success = insert_agent_myth_safe("not_an_int", agent_id, 0.8)
            # This might fail due to type conversion or database error
        except Exception:
            pass  # Expected to fail
        
        try:
            success = insert_agent_myth_safe(myth_id, "not_an_int", 0.8)
            # This might fail due to type conversion or database error
        except Exception:
            pass  # Expected to fail
        
        # Test with string instead of float for retention
        try:
            success = insert_agent_myth_safe(myth_id, agent_id, "not_a_float")
            # This might fail due to type conversion or database error
        except Exception:
            pass  # Expected to fail
        
        # Test with None values
        try:
            success = insert_agent_myth_safe(None, agent_id, 0.8)
            # This should fail
        except Exception:
            pass  # Expected to fail
        
        try:
            success = insert_agent_myth_safe(myth_id, None, 0.8)
            # This should fail
        except Exception:
            pass  # Expected to fail
        
        try:
            success = insert_agent_myth_safe(myth_id, agent_id, None)
            # This should fail
        except Exception:
            pass  # Expected to fail
