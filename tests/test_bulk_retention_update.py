"""
Tests for bulk retention update functionality.
"""

import pytest
import numpy as np
from typing import List, Tuple
from mythologizer_postgres.connectors import (
    update_retentions_and_reorder,
    get_myth_ids_and_retention_from_agents_memory,
    insert_agent_myth,
    get_agent_myth,
    recalculate_agent_myth_positions_by_retention,
)
from mythologizer_postgres.connectors.mythicalgebra import update_myth_with_retention
from mythologizer_postgres.db import psycopg_connection


def get_embedding_dim():
    """Get the embedding dimension from environment variable."""
    import os
    return int(os.getenv('EMBEDDING_DIM', '4'))


class TestBulkRetentionUpdate:
    """Test bulk retention update functionality."""
    
    def _create_test_agent(self) -> int:
        """Create a test agent and return its ID."""
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agents (name, memory_size)
                    VALUES ('Test Agent', 10)
                    RETURNING id
                """)
                agent_id = cur.fetchone()[0]
                conn.commit()
                return agent_id
    
    def _create_test_myth(self) -> int:
        """Create a test myth and return its ID."""
        import numpy as np
        embedding_dim = get_embedding_dim()
        embedding = np.random.rand(embedding_dim).tolist()
        
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights)
                    VALUES (%s, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
                    RETURNING id
                """, (embedding,))
                myth_id = cur.fetchone()[0]
                conn.commit()
                return myth_id
    
    def _insert_agent_myth(self, agent_id: int, myth_id: int, position: int, retention: float) -> bool:
        """Insert an agent_myth entry."""
        return insert_agent_myth(myth_id, agent_id, position, retention)
    
    def _get_agent_myths_ordered(self, agent_id: int) -> List[Tuple[int, float, int]]:
        """Get agent myths ordered by position with myth_id, retention, and position."""
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT myth_id, retention, position
                    FROM agent_myths
                    WHERE agent_id = %s
                    ORDER BY position ASC
                """, (agent_id,))
                return cur.fetchall()
    
    def test_update_retentions_and_reorder_basic(self):
        """Test basic functionality of updating multiple retentions and reordering."""
        agent_id = self._create_test_agent()
        
        # Create 3 test myths
        myth_ids = [self._create_test_myth() for _ in range(3)]
        
        # Insert myths with initial retentions (let triggers handle position assignment)
        self._insert_agent_myth(agent_id, myth_ids[0], 0, 0.3)  # will be assigned position 0
        self._insert_agent_myth(agent_id, myth_ids[1], 0, 0.5)  # will be assigned position 0, pushing myth_ids[0] to position 1
        self._insert_agent_myth(agent_id, myth_ids[2], 0, 0.1)  # will be assigned position 0, pushing others down
        
        # Verify initial order (stack behavior: last inserted at position 0)
        initial_order = self._get_agent_myths_ordered(agent_id)
        assert len(initial_order) == 3
        assert initial_order[0][0] == myth_ids[2]  # myth_ids[2] at position 0 (last inserted)
        assert initial_order[1][0] == myth_ids[1]  # myth_ids[1] at position 1
        assert initial_order[2][0] == myth_ids[0]  # myth_ids[0] at position 2
        
        # Update retentions: myth_ids[2] gets highest retention, myth_ids[0] gets lowest
        myth_retention_pairs = [
            (myth_ids[0], 0.1),  # lowest retention
            (myth_ids[1], 0.3),  # medium retention
            (myth_ids[2], 0.9),  # highest retention
        ]
        
        # Update retentions and trigger reordering
        success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
        assert success is True
        
        # Verify new order: highest retention should be at position 0
        final_order = self._get_agent_myths_ordered(agent_id)
        assert len(final_order) == 3
        
        # myth_ids[2] should be at position 0 (highest retention 0.9)
        assert final_order[0][0] == myth_ids[2]
        assert final_order[0][1] == 0.9  # retention
        assert final_order[0][2] == 0    # position
        
        # myth_ids[1] should be at position 1 (medium retention 0.3)
        assert final_order[1][0] == myth_ids[1]
        assert final_order[1][1] == 0.3  # retention
        assert final_order[1][2] == 1    # position
        
        # myth_ids[0] should be at position 2 (lowest retention 0.1)
        assert final_order[2][0] == myth_ids[0]
        assert final_order[2][1] == 0.1  # retention
        assert final_order[2][2] == 2    # position
    
    def test_update_retentions_and_reorder_tie_breaking(self):
        """Test that tie-breaking works correctly when retentions are equal."""
        agent_id = self._create_test_agent()
        
        # Create 3 test myths
        myth_ids = [self._create_test_myth() for _ in range(3)]
        
        # Insert myths with initial positions
        self._insert_agent_myth(agent_id, myth_ids[0], 0, 0.5)
        self._insert_agent_myth(agent_id, myth_ids[1], 1, 0.5)
        self._insert_agent_myth(agent_id, myth_ids[2], 2, 0.5)
        
        # Update retentions: two myths get same retention, one gets different
        myth_retention_pairs = [
            (myth_ids[0], 0.8),  # highest retention
            (myth_ids[1], 0.8),  # same retention as myth_ids[0] (tie)
            (myth_ids[2], 0.3),  # lowest retention
        ]
        
        # Update retentions and trigger reordering
        success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
        assert success is True
        
        # Verify new order: tie should be broken by myth_id ASC
        final_order = self._get_agent_myths_ordered(agent_id)
        assert len(final_order) == 3
        
        # myth_ids[0] should be at position 0 (same retention as myth_ids[1], but lower myth_id)
        assert final_order[0][0] == myth_ids[0]
        assert final_order[0][1] == 0.8  # retention
        assert final_order[0][2] == 0    # position
        
        # myth_ids[1] should be at position 1 (same retention as myth_ids[0], but higher myth_id)
        assert final_order[1][0] == myth_ids[1]
        assert final_order[1][1] == 0.8  # retention
        assert final_order[1][2] == 1    # position
        
        # myth_ids[2] should be at position 2 (lowest retention)
        assert final_order[2][0] == myth_ids[2]
        assert final_order[2][1] == 0.3  # retention
        assert final_order[2][2] == 2    # position
    
    def test_update_retentions_and_reorder_empty_list(self):
        """Test that empty list of updates returns True."""
        agent_id = self._create_test_agent()
        
        # Test with empty list
        success = update_retentions_and_reorder(agent_id, [])
        assert success is True
        
        # Verify no myths exist
        myths = self._get_agent_myths_ordered(agent_id)
        assert len(myths) == 0
    
    def test_update_retentions_and_reorder_myth_not_found(self):
        """Test that function returns False when myth is not found for agent."""
        agent_id = self._create_test_agent()
        other_agent_id = self._create_test_agent()
        myth_id = self._create_test_myth()
        
        # Insert myth for other agent
        self._insert_agent_myth(other_agent_id, myth_id, 0, 0.5)
        
        # Try to update retention for myth that doesn't belong to this agent
        myth_retention_pairs = [(myth_id, 0.8)]
        
        success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
        assert success is False
        
        # Verify no changes were made
        myths = self._get_agent_myths_ordered(agent_id)
        assert len(myths) == 0
        
        # Verify other agent's myth is unchanged
        other_myths = self._get_agent_myths_ordered(other_agent_id)
        assert len(other_myths) == 1
        assert other_myths[0][1] == 0.5  # retention unchanged
    
    def test_update_retentions_and_reorder_partial_failure(self):
        """Test that function returns False when any myth is not found."""
        agent_id = self._create_test_agent()
        other_agent_id = self._create_test_agent()
        
        # Create myths
        myth_id1 = self._create_test_myth()
        myth_id2 = self._create_test_myth()
        myth_id3 = self._create_test_myth()
        
        # Insert myths for agent (let triggers handle position assignment)
        self._insert_agent_myth(agent_id, myth_id1, 0, 0.5)  # position 0
        self._insert_agent_myth(agent_id, myth_id2, 0, 0.3)  # position 0, myth_id1 moves to position 1
        
        # Insert myth_id3 for other agent
        self._insert_agent_myth(other_agent_id, myth_id3, 0, 0.7)
        
        # Try to update retentions including a myth that doesn't belong to this agent
        myth_retention_pairs = [
            (myth_id1, 0.8),  # valid
            (myth_id2, 0.6),  # valid
            (myth_id3, 0.9),  # invalid - belongs to other agent
        ]
        
        success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
        assert success is False
        
        # Verify no changes were made to this agent
        myths = self._get_agent_myths_ordered(agent_id)
        assert len(myths) == 2
        # After insertion, myth_id2 is at position 0, myth_id1 is at position 1
        assert myths[0][1] == 0.3  # myth_id2 retention unchanged
        assert myths[1][1] == 0.5  # myth_id1 retention unchanged
    
    def test_update_retentions_and_reorder_large_number(self):
        """Test with a larger number of myths to ensure performance."""
        agent_id = self._create_test_agent()
        
        # Create 10 test myths
        myth_ids = [self._create_test_myth() for _ in range(10)]
        
        # Insert myths with initial retentions
        for i, myth_id in enumerate(myth_ids):
            self._insert_agent_myth(agent_id, myth_id, i, 0.1 + (i * 0.1))
        
        # Create retention pairs with reverse order (highest retention first)
        myth_retention_pairs = [
            (myth_ids[i], 1.0 - (i * 0.1)) for i in range(10)
        ]
        
        # Update retentions and trigger reordering
        success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
        assert success is True
        
        # Verify new order: highest retention should be at position 0
        final_order = self._get_agent_myths_ordered(agent_id)
        assert len(final_order) == 10
        
        # Check that positions are 0-indexed and ordered by retention DESC
        for i, (myth_id, retention, position) in enumerate(final_order):
            assert position == i  # 0-indexed positions
            expected_retention = 1.0 - (i * 0.1)
            assert retention == pytest.approx(expected_retention, rel=1e-9)
    
    def test_update_retentions_and_reorder_with_memory_store_function(self):
        """Test that the function works correctly with the memory store function."""
        agent_id = self._create_test_agent()
        
        # Create 3 test myths
        myth_ids = [self._create_test_myth() for _ in range(3)]
        
        # Insert myths with initial retentions (let triggers handle position assignment)
        self._insert_agent_myth(agent_id, myth_ids[0], 0, 0.3)  # position 0
        self._insert_agent_myth(agent_id, myth_ids[1], 0, 0.5)  # position 0, myth_ids[0] moves to position 1
        self._insert_agent_myth(agent_id, myth_ids[2], 0, 0.1)  # position 0, others move down
        
        # Get initial state using memory store function
        initial_myth_ids, initial_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        # Expected order: myth_ids[2] (last inserted), myth_ids[1], myth_ids[0]
        expected_myth_ids = [myth_ids[2], myth_ids[1], myth_ids[0]]
        expected_retentions = [0.1, 0.5, 0.3]
        assert initial_myth_ids == expected_myth_ids
        assert initial_retentions == expected_retentions
        
        # Update retentions
        myth_retention_pairs = [
            (myth_ids[0], 0.1),  # lowest retention
            (myth_ids[1], 0.3),  # medium retention
            (myth_ids[2], 0.9),  # highest retention
        ]
        
        success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
        assert success is True
        
        # Get final state using memory store function
        final_myth_ids, final_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        # Verify order: highest retention should be first
        assert final_myth_ids == [myth_ids[2], myth_ids[1], myth_ids[0]]
        assert final_retentions == [0.9, 0.3, 0.1]
    
    def test_update_retentions_and_reorder_edge_case_identical_retentions(self):
        """Test edge case where all retentions are identical."""
        agent_id = self._create_test_agent()
        
        # Create 3 test myths
        myth_ids = [self._create_test_myth() for _ in range(3)]
        
        # Insert myths with different initial retentions
        self._insert_agent_myth(agent_id, myth_ids[0], 0, 0.3)
        self._insert_agent_myth(agent_id, myth_ids[1], 1, 0.5)
        self._insert_agent_myth(agent_id, myth_ids[2], 2, 0.1)
        
        # Update all to same retention
        myth_retention_pairs = [
            (myth_ids[0], 0.5),
            (myth_ids[1], 0.5),
            (myth_ids[2], 0.5),
        ]
        
        success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
        assert success is True
        
        # Verify order: should be ordered by myth_id ASC due to tie-breaking
        final_order = self._get_agent_myths_ordered(agent_id)
        assert len(final_order) == 3
        
        # All should have same retention
        for myth_id, retention, position in final_order:
            assert retention == 0.5
        
        # Should be ordered by myth_id ASC
        assert final_order[0][0] == myth_ids[0]  # lowest myth_id
        assert final_order[1][0] == myth_ids[1]  # medium myth_id
        assert final_order[2][0] == myth_ids[2]  # highest myth_id
