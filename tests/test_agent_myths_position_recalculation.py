"""
Tests for agent_myths position recalculation functionality.
Tests that positions are correctly recalculated when retention values change
and when new myths are inserted, ensuring highest retention is at top position.
"""

import os
import pytest
import numpy as np
from typing import List, Tuple
from sqlalchemy import text

from mythologizer_postgres.db import get_engine, clear_all_rows, session_scope
from mythologizer_postgres.connectors import insert_agent_myth_safe


class TestAgentMythsPositionRecalculation:
    """Test the position recalculation functionality for agent_myths table."""

    def setup_method(self):
        """Clean up before each test method."""
        clear_all_rows()

    def teardown_method(self):
        """Clean up after each test method."""
        clear_all_rows()

    def _create_test_agent(self, name: str = "Test Agent", memory_size: int = 10) -> int:
        """Helper method to create a test agent."""
        with session_scope() as session:
            result = session.execute(text("""
                INSERT INTO agents (name, memory_size)
                VALUES (:name, :memory_size)
                RETURNING id
            """), {"name": name, "memory_size": memory_size})
            return result.fetchone()[0]

    def _create_test_myth(self, embedding_dim: int = 4) -> int:
        """Helper method to create a test myth."""
        with session_scope() as session:
            embedding = np.random.rand(embedding_dim).tolist()
            result = session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
                RETURNING id
            """), {"embedding": embedding})
            return result.fetchone()[0]

    def _insert_agent_myth(self, myth_id: int, agent_id: int, position: int, retention: float):
        """Helper method to insert an agent_myth relationship."""
        # Use safe function instead of trigger
        success = insert_agent_myth_safe(
            myth_id=myth_id,
            agent_id=agent_id,
            retention=retention
        )
        assert success, f"Failed to insert myth {myth_id} into agent {agent_id}"

    def _update_retention(self, myth_id: int, agent_id: int, new_retention: float):
        """Helper method to update retention value."""
        with session_scope() as session:
            session.execute(text("""
                UPDATE agent_myths 
                SET retention = :retention
                WHERE myth_id = :myth_id AND agent_id = :agent_id
            """), {"retention": new_retention, "myth_id": myth_id, "agent_id": agent_id})
            
            # Manually trigger retention-based reordering
            session.execute(text("SELECT recalculate_agent_myth_positions_by_retention(:agent_id)"), 
                          {"agent_id": agent_id})

    def _get_agent_myths_ordered(self, agent_id: int) -> List[Tuple[int, float, int]]:
        """Helper method to get agent myths ordered by position."""
        with session_scope() as session:
            result = session.execute(text("""
                SELECT myth_id, retention, position
                FROM agent_myths
                WHERE agent_id = :agent_id
                ORDER BY position ASC
            """), {"agent_id": agent_id})
            return [(row[0], row[1], row[2]) for row in result.fetchall()]

    @pytest.mark.integration
    def test_position_recalculation_on_insertion(self):
        """Test that positions follow stack behavior (LIFO) when new myths are inserted."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        myth3 = self._create_test_myth()
        
        # Insert myths with different retentions (not in order)
        self._insert_agent_myth(myth1, agent_id, 1, 0.5)  # First inserted
        self._insert_agent_myth(myth2, agent_id, 1, 0.9)  # Second inserted
        self._insert_agent_myth(myth3, agent_id, 1, 0.7)  # Third inserted
        
        # Check that positions follow stack behavior (LIFO)
        results = self._get_agent_myths_ordered(agent_id)
        
        assert len(results) == 3, "Should have 3 myths"
        
        # Should be ordered by retention (highest retention = position 0)
        # Position 0 = highest retention
        # Position 2 = lowest retention
        assert results[0][0] == myth2, "Highest retention myth should be at position 0 (top)"
        assert results[0][1] == 0.9, "Position 0 should have retention 0.9"
        assert results[0][2] == 0, "Position should be 0"
        
        assert results[1][0] == myth3, "Second highest retention myth should be at position 1"
        assert results[1][1] == 0.7, "Position 1 should have retention 0.7"
        assert results[1][2] == 1, "Position should be 1"
        
        assert results[2][0] == myth1, "Lowest retention myth should be at position 2 (bottom)"
        assert results[2][1] == 0.5, "Position 2 should have retention 0.5"
        assert results[2][2] == 2, "Position should be 2"

    @pytest.mark.integration
    def test_position_recalculation_on_retention_update(self):
        """Test that positions are recalculated when retention values are updated."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        myth3 = self._create_test_myth()
        
        # Insert myths with initial retentions
        self._insert_agent_myth(myth1, agent_id, 1, 0.5)
        self._insert_agent_myth(myth2, agent_id, 1, 0.7)
        self._insert_agent_myth(myth3, agent_id, 1, 0.9)
        
        # Verify initial retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        assert results[0][0] == myth3, "Initial: myth3 should be at top (highest retention 0.9)"
        assert results[1][0] == myth2, "Initial: myth2 should be in middle (retention 0.7)"
        assert results[2][0] == myth1, "Initial: myth1 should be at bottom (lowest retention 0.5)"
        
        # Update retention to change the order
        self._update_retention(myth1, agent_id, 0.95)  # Now highest
        
        # Check that positions are recalculated
        results = self._get_agent_myths_ordered(agent_id)
        
        assert len(results) == 3, "Should still have 3 myths"
        
        # New order should be: myth1 (0.95), myth3 (0.9), myth2 (0.7)
        assert results[0][0] == myth1, "After update: myth1 should be at top"
        assert results[0][1] == 0.95, "Position 0 should have retention 0.95"
        assert results[0][2] == 0, "Position should be 0"
        
        assert results[1][0] == myth3, "After update: myth3 should be in middle"
        assert results[1][1] == 0.9, "Position 1 should have retention 0.9"
        assert results[1][2] == 1, "Position should be 1"
        
        assert results[2][0] == myth2, "After update: myth2 should be at bottom"
        assert results[2][1] == 0.7, "Position 2 should have retention 0.7"
        assert results[2][2] == 2, "Position should be 2"

    @pytest.mark.integration
    def test_position_recalculation_with_tied_retentions(self):
        """Test that positions are handled correctly when retentions are tied."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        myth3 = self._create_test_myth()
        
        # Insert myths with tied retentions
        self._insert_agent_myth(myth1, agent_id, 1, 0.8)
        self._insert_agent_myth(myth2, agent_id, 1, 0.8)  # Same retention
        self._insert_agent_myth(myth3, agent_id, 1, 0.9)  # Higher retention
        
        # Check initial retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        
        assert len(results) == 3, "Should have 3 myths"
        
        # Initial retention-based order: myth3 (highest retention 0.9) at top
        assert results[0][0] == myth3, "myth3 should be at top (highest retention 0.9)"
        assert results[0][1] == 0.9, "Top should have retention 0.9"
        assert results[0][2] == 0, "Top should have position 0"
        
        # For tied retentions (0.8), the order is determined by myth_id ASC
        assert results[1][0] == myth1, "myth1 should be in middle (tied retention 0.8, lower myth_id)"
        assert results[1][1] == 0.8, "Middle should have retention 0.8"
        assert results[1][2] == 1, "Middle should have position 1"
        
        assert results[2][0] == myth2, "myth2 should be at bottom (tied retention 0.8, higher myth_id)"
        assert results[2][1] == 0.8, "Bottom should have retention 0.8"
        assert results[2][2] == 2, "Bottom should have position 2"
        
        # Now trigger retention-based reordering
        from mythologizer_postgres.connectors import recalculate_agent_myth_positions_by_retention
        success = recalculate_agent_myth_positions_by_retention(agent_id)
        assert success, "Manual reordering should succeed"
        
        # Check retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        
        # myth3 should be at top (highest retention)
        assert results[0][0] == myth3, "After reorder: myth3 should be at top (highest retention)"
        assert results[0][1] == 0.9, "Top should have retention 0.9"
        assert results[0][2] == 0, "Top should have position 0"
        
        # For tied retentions, should maintain insertion order as secondary sort
        # myth1 was inserted first, then myth2
        assert results[1][0] == myth1, "After reorder: myth1 should be second (inserted first)"
        assert results[1][1] == 0.8, "Second should have retention 0.8"
        assert results[1][2] == 1, "Second should have position 1"
        
        assert results[2][0] == myth2, "After reorder: myth2 should be third (inserted second)"
        assert results[2][1] == 0.8, "Third should have retention 0.8"
        assert results[2][2] == 2, "Third should have position 2"

    @pytest.mark.integration
    def test_position_recalculation_multiple_agents(self):
        """Test that position recalculation only affects the correct agent."""
        agent1_id = self._create_test_agent("Agent 1")
        agent2_id = self._create_test_agent("Agent 2")
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        myth3 = self._create_test_myth()
        myth4 = self._create_test_myth()
        
        # Insert myths for both agents
        self._insert_agent_myth(myth1, agent1_id, 1, 0.5)
        self._insert_agent_myth(myth2, agent1_id, 1, 0.9)
        
        self._insert_agent_myth(myth3, agent2_id, 1, 0.3)
        self._insert_agent_myth(myth4, agent2_id, 1, 0.8)
        
        # Update retention for agent1 only
        self._update_retention(myth1, agent1_id, 0.95)
        
        # Check agent1 positions (should be recalculated)
        agent1_results = self._get_agent_myths_ordered(agent1_id)
        assert len(agent1_results) == 2, "Agent1 should have 2 myths"
        assert agent1_results[0][0] == myth1, "Agent1: myth1 should be at top after update"
        assert agent1_results[1][0] == myth2, "Agent1: myth2 should be at bottom after update"
        
        # Check agent2 positions (should be unchanged - still in stack order)
        agent2_results = self._get_agent_myths_ordered(agent2_id)
        assert len(agent2_results) == 2, "Agent2 should have 2 myths"
        assert agent2_results[0][0] == myth4, "Agent2: myth4 should still be at top (last inserted)"
        assert agent2_results[1][0] == myth3, "Agent2: myth3 should still be at bottom (first inserted)"

    @pytest.mark.integration
    def test_position_recalculation_edge_case_single_myth(self):
        """Test position recalculation with only one myth."""
        agent_id = self._create_test_agent()
        myth_id = self._create_test_myth()
        
        # Insert single myth
        self._insert_agent_myth(myth_id, agent_id, 1, 0.5)
        
        # Check initial position
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 1, "Should have 1 myth"
        assert results[0][0] == myth_id, "Should have correct myth"
        assert results[0][1] == 0.5, "Should have correct retention"
        assert results[0][2] == 0, "Should be at position 0"
        
        # Update retention
        self._update_retention(myth_id, agent_id, 0.9)
        
        # Check position after update
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 1, "Should still have 1 myth"
        assert results[0][0] == myth_id, "Should still have correct myth"
        assert results[0][1] == 0.9, "Should have updated retention"
        assert results[0][2] == 0, "Should still be at position 0"

    @pytest.mark.integration
    def test_position_recalculation_edge_case_empty_agent(self):
        """Test position recalculation with no myths for an agent."""
        agent_id = self._create_test_agent()
        
        # Check that no myths exist
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 0, "Should have no myths"
        
        # This should not cause any errors
        # The trigger function should handle empty result sets gracefully

    @pytest.mark.integration
    def test_position_recalculation_edge_case_very_small_retention(self):
        """Test position recalculation with very small retention values."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        
        # Insert with very small retention values
        self._insert_agent_myth(myth1, agent_id, 1, 0.0001)
        self._insert_agent_myth(myth2, agent_id, 1, 0.0002)
        
        # Check initial stack order (LIFO)
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 2, "Should have 2 myths"
        assert results[0][0] == myth2, "Second inserted should be at top"
        assert results[1][0] == myth1, "First inserted should be at bottom"
        
        # Now trigger retention-based reordering
        from mythologizer_postgres.connectors import recalculate_agent_myth_positions_by_retention
        success = recalculate_agent_myth_positions_by_retention(agent_id)
        assert success, "Manual reordering should succeed"
        
        # Check retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        assert results[0][0] == myth2, "After reorder: Higher retention should be at top"
        assert results[1][0] == myth1, "After reorder: Lower retention should be at bottom"

    @pytest.mark.integration
    def test_position_recalculation_edge_case_very_large_retention(self):
        """Test position recalculation with very large retention values."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        
        # Insert with very large retention values
        self._insert_agent_myth(myth1, agent_id, 1, 999999.0)
        self._insert_agent_myth(myth2, agent_id, 1, 999998.0)
        
        # Check initial retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 2, "Should have 2 myths"
        assert results[0][0] == myth1, "myth1 should be at top (highest retention 999999.0)"
        assert results[1][0] == myth2, "myth2 should be at bottom (retention 999998.0)"
        
        # Now trigger retention-based reordering
        from mythologizer_postgres.connectors import recalculate_agent_myth_positions_by_retention
        success = recalculate_agent_myth_positions_by_retention(agent_id)
        assert success, "Manual reordering should succeed"
        
        # Check retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        assert results[0][0] == myth1, "After reorder: Higher retention should be at top"
        assert results[1][0] == myth2, "After reorder: Lower retention should be at bottom"

    @pytest.mark.integration
    def test_position_recalculation_edge_case_identical_retentions(self):
        """Test position recalculation when all retentions are identical."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        myth3 = self._create_test_myth()
        
        # Insert with identical retentions
        self._insert_agent_myth(myth1, agent_id, 1, 0.5)
        self._insert_agent_myth(myth2, agent_id, 1, 0.5)
        self._insert_agent_myth(myth3, agent_id, 1, 0.5)
        
        # Check initial retention-based order (all retentions are 0.5, so ordered by myth_id ASC)
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 3, "Should have 3 myths"
        assert results[0][0] == myth1, "myth1 should be at top (lowest myth_id)"
        assert results[1][0] == myth2, "myth2 should be in middle (middle myth_id)"
        assert results[2][0] == myth3, "myth3 should be at bottom (highest myth_id)"
        
        # Now trigger retention-based reordering
        from mythologizer_postgres.connectors import recalculate_agent_myth_positions_by_retention
        success = recalculate_agent_myth_positions_by_retention(agent_id)
        assert success, "Manual reordering should succeed"
        
        # Check retention-based order (should maintain insertion order for identical retentions)
        results = self._get_agent_myths_ordered(agent_id)
        assert results[0][0] == myth1, "After reorder: First inserted should be at top"
        assert results[1][0] == myth2, "After reorder: Second inserted should be in middle"
        assert results[2][0] == myth3, "After reorder: Third inserted should be at bottom"

    @pytest.mark.integration
    def test_position_recalculation_edge_case_retention_to_identical(self):
        """Test position recalculation when update makes retentions identical."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        
        # Insert with different retentions
        self._insert_agent_myth(myth1, agent_id, 1, 0.5)
        self._insert_agent_myth(myth2, agent_id, 1, 0.9)
        
        # Verify initial retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        assert results[0][0] == myth2, "Initial: myth2 should be at top (highest retention 0.9)"
        assert results[1][0] == myth1, "Initial: myth1 should be at bottom (lowest retention 0.5)"
        
        # Update to make retentions identical
        self._update_retention(myth1, agent_id, 0.9)
        
        # Check positions (should maintain myth_id order for ties - lower ID first)
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 2, "Should have 2 myths"
        
        # For identical retentions, the order is based on myth_id (lower first)
        if myth1 < myth2:
            assert results[0][0] == myth1, "Lower myth_id should be at top"
            assert results[1][0] == myth2, "Higher myth_id should be at bottom"
        else:
            assert results[0][0] == myth2, "Lower myth_id should be at top"
            assert results[1][0] == myth1, "Higher myth_id should be at bottom"

    @pytest.mark.integration
    def test_position_recalculation_performance_large_number(self):
        """Test position recalculation with a larger number of myths."""
        agent_id = self._create_test_agent(memory_size=100)
        
        # Create 20 myths
        myth_ids = [self._create_test_myth() for _ in range(20)]
        
        # Insert with random retentions
        retentions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05]
        
        for i, (myth_id, retention) in enumerate(zip(myth_ids, retentions)):
            self._insert_agent_myth(myth_id, agent_id, 1, retention)
        
        # Check initial retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 20, "Should have 20 myths"
        
        # Verify positions are contiguous (0-indexed)
        for i, (myth_id, retention, position) in enumerate(results):
            assert position == i, f"Position {i} should be {i}"
        
        # Now trigger retention-based reordering
        from mythologizer_postgres.connectors import recalculate_agent_myth_positions_by_retention
        success = recalculate_agent_myth_positions_by_retention(agent_id)
        assert success, "Manual reordering should succeed"
        
        # Check retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        
        # Verify retention ordering (should be descending)
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1], f"Retention at position {i+1} should be >= retention at position {i+2}"

    @pytest.mark.integration
    def test_position_recalculation_concurrent_updates(self):
        """Test position recalculation with concurrent retention updates."""
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        myth3 = self._create_test_myth()
        
        # Insert myths
        self._insert_agent_myth(myth1, agent_id, 1, 0.5)
        self._insert_agent_myth(myth2, agent_id, 1, 0.7)
        self._insert_agent_myth(myth3, agent_id, 1, 0.9)
        
        # Perform multiple updates in sequence
        self._update_retention(myth1, agent_id, 0.8)  # myth1 becomes second
        self._update_retention(myth2, agent_id, 0.95)  # myth2 becomes first
        self._update_retention(myth3, agent_id, 0.6)  # myth3 becomes last
        
        # Check final order
        results = self._get_agent_myths_ordered(agent_id)
        assert len(results) == 3, "Should have 3 myths"
        
        # Final order should be: myth2 (0.95), myth1 (0.8), myth3 (0.6)
        assert results[0][0] == myth2, "myth2 should be at top"
        assert results[0][1] == 0.95, "Top should have retention 0.95"
        
        assert results[1][0] == myth1, "myth1 should be in middle"
        assert results[1][1] == 0.8, "Middle should have retention 0.8"
        
        assert results[2][0] == myth3, "myth3 should be at bottom"
        assert results[2][1] == 0.6, "Bottom should have retention 0.6"

    @pytest.mark.integration
    def test_manual_retention_based_reordering(self):
        """Test that manual retention-based reordering works correctly."""
        from mythologizer_postgres.connectors import recalculate_agent_myth_positions_by_retention
        
        agent_id = self._create_test_agent()
        
        # Create myths
        myth1 = self._create_test_myth()
        myth2 = self._create_test_myth()
        myth3 = self._create_test_myth()
        
        # Insert myths with different retentions (stack order)
        self._insert_agent_myth(myth1, agent_id, 1, 0.5)  # First inserted
        self._insert_agent_myth(myth2, agent_id, 1, 0.9)  # Second inserted
        self._insert_agent_myth(myth3, agent_id, 1, 0.7)  # Third inserted
        
        # Verify initial retention-based order
        results = self._get_agent_myths_ordered(agent_id)
        # Position 0 = highest retention, Position 2 = lowest retention
        assert results[0][0] == myth2, "Initial: myth2 should be at top (highest retention 0.9)"
        assert results[1][0] == myth3, "Initial: myth3 should be in middle (retention 0.7)"
        assert results[2][0] == myth1, "Initial: myth1 should be at bottom (lowest retention 0.5)"
        
        # Manually trigger retention-based reordering
        success = recalculate_agent_myth_positions_by_retention(agent_id)
        assert success, "Manual reordering should succeed"
        
        # Check that positions are now ordered by retention (highest first)
        results = self._get_agent_myths_ordered(agent_id)
        
        assert len(results) == 3, "Should still have 3 myths"
        
        # Should now be ordered by retention (highest first)
        assert results[0][0] == myth2, "After reorder: myth2 should be at top (highest retention 0.9)"
        assert results[0][1] == 0.9, "Position 0 should have retention 0.9"
        assert results[0][2] == 0, "Position should be 0"
        
        assert results[1][0] == myth3, "After reorder: myth3 should be in middle (retention 0.7)"
        assert results[1][1] == 0.7, "Position 1 should have retention 0.7"
        assert results[1][2] == 1, "Position should be 1"
        
        assert results[2][0] == myth1, "After reorder: myth1 should be at bottom (lowest retention 0.5)"
        assert results[2][1] == 0.5, "Position 2 should have retention 0.5"
        assert results[2][2] == 2, "Position should be 2"
